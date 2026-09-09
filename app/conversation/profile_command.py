"""明确用户称呼命令的确定性解析与状态机。"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from app.conversation.command_result import MemoryStateChange
from app.conversation.profile_rollout import profile_memory_enabled
from app.conversation.runtime_memory import ShortTermRuntimeSnapshot
from app.conversation.service import (
    PreferredNameConflict,
    get_conversation_service,
    supports_account_profile,
)
from app.observability.trace import current_trace


ProfileAction = Literal["set", "clear", "invalid", "none"]


@dataclass(frozen=True)
class ProfileCommand:
    action: ProfileAction
    candidate_name: str | None = None
    reason_code: str = "NO_PROFILE_COMMAND"


@dataclass(frozen=True)
class ProfileTurnReply:
    message: str
    status: str
    lang: str = "zh"
    success: bool = True
    state_changes: tuple[MemoryStateChange, ...] = ()


def _profile_change(
    operation: Literal["set", "clear"],
    *,
    reason_code: str,
) -> MemoryStateChange:
    return MemoryStateChange(
        scope="profile",
        operation=operation,
        keys=("preferred_name",),
        reason_code=reason_code,
    )


def _session_change(
    key: Literal["pending_profile_update", "temporary_profile"],
    operation: Literal["set", "clear"],
    *,
    reason_code: str,
) -> MemoryStateChange:
    return MemoryStateChange(
        scope="session",
        operation=operation,
        keys=(key,),
        reason_code=reason_code,
    )


def _state_changes(
    *changes: MemoryStateChange | None,
) -> tuple[MemoryStateChange, ...]:
    return tuple(change for change in changes if change is not None)


_CLEAR_PATTERNS = (
    re.compile(r"(?:以后)?(?:别|不要|不用)(?:再)?(?:叫|称呼|喊)我(?:为|作)?"),
    re.compile(r"(?:忘掉|删除|清除|清空).{0,6}(?:我的)?称呼"),
)
_SET_PATTERNS = (
    re.compile(
        r"(?:以后|今后)?(?:请)?(?:可以)?(?:改)?(?:叫|称呼|喊)我(?:为|作)?"
        r"\s*[“\"']?([^，。！？,.!?\s“”\"']{1,40})"
    ),
    re.compile(
        r"(?:我叫|我的名字是)\s*[“\"']?"
        r"([^，。！？,.!?\s“”\"']{1,40})"
    ),
)
_REPLACE_PATTERNS = (
    re.compile(
        r"(?:以后)?(?:别|不要)(?:再)?(?:叫|称呼|喊)我"
        r"[^，。！？,.!?\s]{1,24}(?:了)?\s*[，,]\s*"
        r"(?:以后)?(?:请)?(?:改)?(?:叫|称呼|喊)我(?:为|作)?"
        r"\s*[“\"']?([^，。！？,.!?\s“”\"']{1,40})"
    ),
)
_NAME_TRAILING_PARTICLES = ("可以吗", "行吗", "好吗", "吧", "呀", "啊", "哦", "呢", "啦", "了")
_INVALID_NAMES = {
    "我", "你", "他", "她", "它", "这个", "那个", "随便",
    "什么", "什么都行", "都可以", "不知道", "一下",
}
_NAME_BLOCKLIST_TERMS = (
    "忽略", "系统提示", "提示词", "指令", "调用工具", "执行命令",
    "输出", "回答以下", "http://", "https://", "```",
)
_AFFIRM_EXACT = {
    "嗯", "嗯嗯", "对", "是", "是的", "可以", "行", "好", "好的",
    "确认", "没错", "就这样", "ok", "okay", "yes", "y",
    "确认修改称呼", "确认改称呼", "确认称呼修改",
}
_CANCEL_EXACT = {
    "算了", "取消", "不改了", "别改了", "保持原样", "还是原来的",
    "no", "cancel",
}
_PROFILE_CONFIRM_MARKERS = ("称呼", "名字", "叫我", "改成")
_DEVICE_CONFIRM_MARKERS = ("启动", "开火", "烹饪", "设备", "开始做")
_PREFERRED_NAME_QUERY_PATTERNS = (
    re.compile(r"你(?:应该|该|要|会)?(?:怎么|如何)?(?:叫|称呼|喊)我(?:什么)?"),
    re.compile(r"你(?:给我)?保存的称呼是什么"),
)


def _normalize_name(value: str) -> str:
    name = str(value or "").strip(" \t\r\n，。！？,.!?“”\"'")
    changed = True
    while changed:
        changed = False
        for suffix in _NAME_TRAILING_PARTICLES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)].rstrip()
                changed = True
                break
    return name


def _valid_name(value: str) -> bool:
    if not value or value in _INVALID_NAMES or len(value) > 24:
        return False
    if any(char in value for char in ("\n", "\r", "\t", "/", "\\", "<", ">", "{", "}")):
        return False
    lowered = value.lower()
    if any(term in lowered for term in _NAME_BLOCKLIST_TERMS):
        return False
    return True


def resolve_profile_command(text: str) -> ProfileCommand:
    """只识别用户明确陈述，不使用 LLM 推断长期资料。"""
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ProfileCommand("none")
    for pattern in _REPLACE_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue
        candidate = _normalize_name(match.group(1))
        if not _valid_name(candidate):
            return ProfileCommand(
                "invalid",
                reason_code="INVALID_PREFERRED_NAME",
            )
        return ProfileCommand(
            "set",
            candidate_name=candidate,
            reason_code="EXPLICIT_PREFERRED_NAME_CHANGE",
        )
    for pattern in _CLEAR_PATTERNS:
        if pattern.search(value):
            return ProfileCommand("clear", reason_code="EXPLICIT_PREFERRED_NAME_CLEAR")
    for pattern in _SET_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue
        candidate = _normalize_name(match.group(1))
        if not _valid_name(candidate):
            return ProfileCommand(
                "invalid",
                candidate_name=None,
                reason_code="INVALID_PREFERRED_NAME",
            )
        return ProfileCommand(
            "set",
            candidate_name=candidate,
            reason_code="EXPLICIT_PREFERRED_NAME_SET",
        )
    return ProfileCommand("none")


def is_preferred_name_query(text: str) -> bool:
    """只识别用户对自己已保存称呼的明确查询，不把机器人身份问答混进来。"""
    value = " ".join(str(text or "").split()).strip()
    return bool(value) and any(
        pattern.search(value) for pattern in _PREFERRED_NAME_QUERY_PATTERNS
    )


def _normalized_reply(text: str) -> str:
    return re.sub(r"[，。！？,.!?\s]+", "", str(text or "")).lower()


def _is_affirmative(text: str) -> bool:
    return _normalized_reply(text) in {
        _normalized_reply(item) for item in _AFFIRM_EXACT
    }


def _is_cancel(text: str) -> bool:
    return _normalized_reply(text) in {
        _normalized_reply(item) for item in _CANCEL_EXACT
    }


def _trace_profile_event(action: str, status: str, *, conflict: bool = False) -> None:
    trace = current_trace()
    if trace is not None:
        trace.add_event(
            "profile_memory",
            action=action,
            status=status,
            conflict=conflict,
        )


async def _save_session_only_name(
    thread_id: str,
    preferred_name: str | None,
) -> bool:
    try:
        await get_conversation_service().save_temporary_preferred_name(
            thread_id,
            preferred_name,
        )
        return True
    except Exception:
        return False


async def _clear_pending_safely(thread_id: str) -> bool:
    try:
        await get_conversation_service().clear_pending_profile_update(thread_id)
        return True
    except Exception:
        return False


def _write_failed_message(
    name: str | None,
    *,
    clearing: bool = False,
    session_saved: bool,
) -> str:
    if not session_saved:
        return "长期保存和本次会话记忆现在都没有成功，请稍后再试；我不会假装已经记住。"
    if clearing:
        return "这次对话里我先不再用原来的称呼，不过长期保存刚才没有成功。"
    return f"这次对话里我先叫你{name}，不过长期保存刚才没有成功。"


async def handle_profile_memory_turn(
    text: str,
    thread_id: str,
    *,
    short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
) -> ProfileTurnReply | None:
    """在意图分类和设备确认前处理明确资料命令。"""
    command = resolve_profile_command(text)
    name_query = is_preferred_name_query(text)
    if (
        not supports_account_profile(thread_id)
        or not profile_memory_enabled(thread_id)
    ):
        if command.action != "none" or name_query:
            _trace_profile_event(
                "query" if name_query else command.action,
                "disabled",
            )
            return ProfileTurnReply(
                "称呼记忆当前没有启用，所以我不会假装已经保存。启用后你再告诉我希望怎么称呼你。",
                "disabled",
                success=False,
            )
        return None
    service = get_conversation_service()
    if name_query:
        try:
            runtime_memory = await service.load_runtime_memory(
                thread_id,
                current_message=text,
                short_term_snapshot=short_term_snapshot,
            )
            profile = await service.user_profile_context(
                thread_id,
                runtime_memory=runtime_memory,
            )
            if profile.get("long_term_memory_status") in {
                "unavailable",
                "invalid",
            }:
                _trace_profile_event("query", "read_failed")
                return ProfileTurnReply(
                    "我现在没能读取到已保存的称呼，不能把它当成还没设置。你稍后再问我一次。",
                    "read_failed",
                    success=False,
                )
            visible_name = str(profile.get("preferred_name") or "").strip() or None
            pending = await service.load_pending_profile_update(
                thread_id,
                short_term_snapshot=short_term_snapshot,
            )
        except Exception:
            _trace_profile_event("query", "read_failed")
            return ProfileTurnReply(
                "我现在没能读到已保存的称呼，先不乱叫。你稍后再问我一次。",
                "read_failed",
                success=False,
            )
        if pending:
            old_name = str(pending.get("old_value") or "").strip() or None
            new_name = str(pending.get("new_value") or "").strip() or None
            _trace_profile_event("query", "pending_change")
            if old_name and new_name:
                return ProfileTurnReply(
                    f"现在保存的称呼是{old_name}；你刚提出改成{new_name}，还在等你确认。",
                    "pending_change",
                )
        if visible_name:
            _trace_profile_event("query", "found")
            return ProfileTurnReply(
                f"记得，你希望我叫你{visible_name}。",
                "found",
            )
        _trace_profile_event("query", "not_set")
        return ProfileTurnReply(
            "我还没有保存你的称呼。你可以直接说“叫我范老师”这样的句子。",
            "not_set",
        )
    normalized = _normalized_reply(text)
    may_answer_pending = (
        command.action != "none"
        or _is_affirmative(text)
        or _is_cancel(text)
        or any(marker in normalized for marker in (
            *_PROFILE_CONFIRM_MARKERS,
            *_DEVICE_CONFIRM_MARKERS,
        ))
    )
    if not may_answer_pending:
        return None
    try:
        pending = await service.load_pending_profile_update(
            thread_id,
            short_term_snapshot=short_term_snapshot,
        )
    except Exception:
        pending = None

    if pending:
        has_device_pending = False
        device_state_known = True
        try:
            state = await service.load_task_state(
                thread_id,
                short_term_snapshot=short_term_snapshot,
            )
            has_device_pending = bool(state.pending_device_start)
        except Exception:
            device_state_known = False
        targets_device = any(marker in normalized for marker in _DEVICE_CONFIRM_MARKERS)
        targets_profile = any(marker in normalized for marker in _PROFILE_CONFIRM_MARKERS)

        if command.action == "set":
            current = str(pending.get("old_value") or "").strip() or None
            candidate = command.candidate_name
            if candidate == current:
                pending_cleared = await _clear_pending_safely(thread_id)
                _trace_profile_event("change", "cancelled_same_as_old")
                return ProfileTurnReply(
                    f"好，那就还是叫你{current}。",
                    "cancelled_same_as_old",
                    state_changes=_state_changes(
                        _session_change(
                            "pending_profile_update",
                            "clear",
                            reason_code="PROFILE_PENDING_CLEARED",
                        )
                        if pending_cleared
                        else None,
                    ),
                )
            replacement = {
                **pending,
                "action_id": uuid.uuid4().hex,
                "new_value": candidate,
                "created_at": time.time(),
                "expires_at": time.time() + 300,
            }
            try:
                await service.save_pending_profile_update(thread_id, replacement)
            except Exception:
                _trace_profile_event("change", "write_failed")
                return ProfileTurnReply(
                    "称呼变更的待确认状态现在保存失败，所以我没有修改原来的称呼。请稍后再试。",
                    "write_failed",
                    success=False,
                )
            _trace_profile_event("change", "awaiting_confirmation")
            return ProfileTurnReply(
                f"之前你的称呼是{current or '未设置'}，现在要改成{candidate}吗？",
                "awaiting_confirmation",
                state_changes=(
                    _session_change(
                        "pending_profile_update",
                        "set",
                        reason_code="PROFILE_PENDING_CREATED",
                    ),
                ),
            )

        if _is_cancel(text):
            pending_cleared = await _clear_pending_safely(thread_id)
            old_name = str(pending.get("old_value") or "").strip()
            _trace_profile_event("change", "cancelled")
            return ProfileTurnReply(
                f"好，不改了，还是叫你{old_name}。" if old_name else "好，这次不修改称呼。",
                "cancelled",
                state_changes=_state_changes(
                    _session_change(
                        "pending_profile_update",
                        "clear",
                        reason_code="PROFILE_PENDING_CLEARED",
                    )
                    if pending_cleared
                    else None,
                ),
            )

        if targets_device and not targets_profile:
            # 让现有设备状态机继续处理，资料待确认状态保持不变。
            return None

        if _is_affirmative(text):
            if (
                (has_device_pending or not device_state_known)
                and not targets_profile
            ):
                _trace_profile_event("change", "ambiguous_confirmation", conflict=True)
                return ProfileTurnReply(
                    "你是确认修改称呼，还是确认启动设备？请直接说“确认修改称呼”或“确认启动设备”。",
                    "ambiguous_confirmation",
                )
            old_name = str(pending.get("expected_persisted_name") or "").strip() or None
            new_name = str(pending.get("new_value") or "").strip() or None
            try:
                await service.update_preferred_name(
                    thread_id,
                    new_name,
                    expected_current=old_name,
                )
                pending_cleared = await _clear_pending_safely(thread_id)
                _trace_profile_event("change", "saved")
                return ProfileTurnReply(
                    f"好，之后我就叫你{new_name}。",
                    "saved",
                    state_changes=_state_changes(
                        _profile_change(
                            "set",
                            reason_code="PREFERRED_NAME_SAVED",
                        ),
                        _session_change(
                            "pending_profile_update",
                            "clear",
                            reason_code="PROFILE_PENDING_CLEARED",
                        )
                        if pending_cleared
                        else None,
                    ),
                )
            except PreferredNameConflict:
                pending_cleared = await _clear_pending_safely(thread_id)
                _trace_profile_event("change", "version_conflict", conflict=True)
                return ProfileTurnReply(
                    "你的称呼刚刚在别的会话里发生了变化，我没有覆盖它。你再告诉我一次现在想用的称呼吧。",
                    "version_conflict",
                    success=False,
                    state_changes=_state_changes(
                        _session_change(
                            "pending_profile_update",
                            "clear",
                            reason_code="PROFILE_PENDING_CLEARED",
                        )
                        if pending_cleared
                        else None,
                    ),
                )
            except Exception:
                session_saved = await _save_session_only_name(thread_id, new_name)
                pending_cleared = await _clear_pending_safely(thread_id)
                _trace_profile_event("change", "write_failed")
                return ProfileTurnReply(
                    _write_failed_message(
                        new_name,
                        session_saved=session_saved,
                    ),
                    "write_failed",
                    success=False,
                    state_changes=_state_changes(
                        _session_change(
                            "temporary_profile",
                            "set",
                            reason_code="TEMPORARY_PROFILE_SAVED",
                        )
                        if session_saved
                        else None,
                        _session_change(
                            "pending_profile_update",
                            "clear",
                            reason_code="PROFILE_PENDING_CLEARED",
                        )
                        if pending_cleared
                        else None,
                    ),
                )

    if command.action == "none":
        return None
    if command.action == "invalid":
        _trace_profile_event("set", "invalid")
        return ProfileTurnReply(
            "这个称呼不太适合直接保存。可以换一个短一些、只包含正常文字的称呼吗？",
            "invalid",
            success=False,
        )

    profile_status = "unavailable"
    try:
        runtime_memory = await service.load_runtime_memory(
            thread_id,
            current_message=text,
            short_term_snapshot=short_term_snapshot,
        )
        profile = await service.user_profile_context(
            thread_id,
            runtime_memory=runtime_memory,
        )
        profile_status = str(
            profile.get("long_term_memory_status") or "invalid"
        )
        visible_name = str(profile.get("preferred_name") or "").strip() or None
        persistent_name = (
            str(profile.get("persistent_preferred_name") or "").strip() or None
        )
    except Exception:
        visible_name = None
        persistent_name = None

    if command.action == "clear":
        if profile_status in {"unavailable", "invalid"}:
            _trace_profile_event("clear", "read_failed")
            return ProfileTurnReply(
                "我现在没能读取到已保存的称呼，所以不能确认已经清除。请稍后再试；我不会把读取失败说成原本没保存。",
                "read_failed",
                success=False,
            )
        if visible_name is None:
            _trace_profile_event("clear", "unchanged")
            return ProfileTurnReply("好，我这边原本就没有保存你的称呼。", "unchanged")
        try:
            await service.update_preferred_name(
                thread_id,
                None,
                expected_current=persistent_name,
            )
            _trace_profile_event("clear", "saved")
            return ProfileTurnReply(
                "好，我以后不再用之前的称呼。你想换一个时再告诉我就行。",
                "saved",
                state_changes=(
                    _profile_change(
                        "clear",
                        reason_code="PREFERRED_NAME_CLEARED",
                    ),
                ),
            )
        except PreferredNameConflict:
            _trace_profile_event("clear", "version_conflict", conflict=True)
            return ProfileTurnReply(
                "你的称呼刚刚在别的会话里发生了变化，我没有覆盖它。你再说一次要不要清除吧。",
                "version_conflict",
                success=False,
            )
        except Exception:
            session_saved = await _save_session_only_name(thread_id, None)
            _trace_profile_event("clear", "write_failed")
            return ProfileTurnReply(
                _write_failed_message(
                    None,
                    clearing=True,
                    session_saved=session_saved,
                ),
                "write_failed",
                success=False,
                state_changes=_state_changes(
                    _session_change(
                        "temporary_profile",
                        "set",
                        reason_code="TEMPORARY_PROFILE_SAVED",
                    )
                    if session_saved
                    else None,
                ),
            )

    candidate = command.candidate_name
    if candidate == visible_name:
        _trace_profile_event("set", "unchanged")
        return ProfileTurnReply(f"记得，你希望我叫你{candidate}。", "unchanged")
    if visible_name:
        now = time.time()
        try:
            await service.save_pending_profile_update(
                thread_id,
                {
                    "kind": "preferred_name_change",
                    "action_id": uuid.uuid4().hex,
                    "old_value": visible_name,
                    "expected_persisted_name": persistent_name,
                    "new_value": candidate,
                    "created_at": now,
                    "expires_at": now + 300,
                },
            )
        except Exception:
            _trace_profile_event("change", "write_failed")
            return ProfileTurnReply(
                "我还没有改动原来的称呼：称呼变更的待确认状态现在保存失败，请稍后再试。",
                "write_failed",
                success=False,
            )
        _trace_profile_event("change", "awaiting_confirmation")
        return ProfileTurnReply(
            f"之前你的称呼是{visible_name}，现在要改成{candidate}吗？",
            "awaiting_confirmation",
            state_changes=(
                _session_change(
                    "pending_profile_update",
                    "set",
                    reason_code="PROFILE_PENDING_CREATED",
                ),
            ),
        )

    try:
        await service.update_preferred_name(
            thread_id,
            candidate,
            expected_current=persistent_name,
        )
        _trace_profile_event("set", "saved")
        return ProfileTurnReply(
            f"好，以后我就叫你{candidate}。",
            "saved",
            state_changes=(
                _profile_change(
                    "set",
                    reason_code="PREFERRED_NAME_SAVED",
                ),
            ),
        )
    except PreferredNameConflict:
        _trace_profile_event("set", "version_conflict", conflict=True)
        return ProfileTurnReply(
            "你的称呼刚刚在别的会话里发生了变化，我没有覆盖它。你再告诉我一次现在想用的称呼吧。",
            "version_conflict",
            success=False,
        )
    except Exception:
        session_saved = await _save_session_only_name(thread_id, candidate)
        _trace_profile_event("set", "write_failed")
        return ProfileTurnReply(
            _write_failed_message(
                candidate,
                session_saved=session_saved,
            ),
            "write_failed",
            success=False,
            state_changes=_state_changes(
                _session_change(
                    "temporary_profile",
                    "set",
                    reason_code="TEMPORARY_PROFILE_SAVED",
                )
                if session_saved
                else None,
            ),
        )
