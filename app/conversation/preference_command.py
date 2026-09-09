"""共享通道的显式饮食偏好更新与查询入口。"""
from __future__ import annotations

from dataclasses import dataclass

from app.conversation.command_result import MemoryStateChange
from app.conversation.preference_parser import (
    PreferenceMutation,
    extract_preference_mutations,
    has_additional_food_task,
    is_preference_memory_query,
)
from app.conversation.runtime_memory import ShortTermRuntimeSnapshot
from app.conversation.service import (
    conversation_identity,
    get_conversation_service,
    supports_account_profile,
)
from app.domain.language import detect_lang


@dataclass(frozen=True)
class PreferenceTurnReply:
    message: str
    lang: str
    continue_routing: bool = False
    status: str = "handled"
    success: bool = True
    state_changes: tuple[MemoryStateChange, ...] = ()


def _joined(values: list[str], lang: str) -> str:
    clean = list(dict.fromkeys(
        str(value or "").strip() for value in values if str(value or "").strip()
    ))
    return ", ".join(clean) if lang == "en" else "、".join(clean)


def _preference_summary(context: dict, lang: str) -> str:
    preferences = context.get("preferences") or {}
    temporal = [
        str(item.get("value") or "").strip()
        for item in context.get("temporal_dietary_constraints") or []
        if isinstance(item, dict) and str(item.get("value") or "").strip()
    ]
    groups = [
        ("喜欢", "likes", "Likes"),
        ("不喜欢或不吃", "dislikes", "Dislikes or avoids"),
        ("过敏", "allergens", "Allergies"),
        ("长期饮食要求", "dietary_constraints", "Dietary requirements"),
    ]
    lines: list[str] = []
    for zh_label, key, en_label in groups:
        values = list(preferences.get(key) or [])
        if values:
            label = en_label if lang == "en" else zh_label
            punctuation = ": " if lang == "en" else "："
            lines.append(f"{label}{punctuation}{_joined(values, lang)}")
    if temporal:
        label = "Temporary goals" if lang == "en" else "阶段性目标"
        punctuation = ": " if lang == "en" else "："
        lines.append(f"{label}{punctuation}{_joined(temporal, lang)}")
    if not lines:
        return (
            "I don't currently have any confirmed food preferences saved for you. "
            "Tell me what you like, avoid, or are allergic to whenever you want."
            if lang == "en"
            else
            "目前没有查到你明确告诉过我的饮食偏好。你随时可以告诉我喜欢什么、不吃什么或对什么过敏。"
        )
    prefix = (
        "Here is what I currently use when helping you choose recipes:"
        if lang == "en"
        else "我现在会把这些作为后续选菜参考："
    )
    separator = "; " if lang == "en" else "；"
    return f"{prefix}{separator.join(lines)}。"


def _mutation_ack(
    mutations: list[PreferenceMutation],
    *,
    changed: list[PreferenceMutation],
    lang: str,
    saved: bool,
) -> str:
    effective = changed or mutations
    allergens = [
        item.value for item in effective
        if item.operation == "add" and item.bucket == "allergens"
    ]
    dislikes = [
        item.value for item in effective
        if (
            item.operation == "add"
            and item.bucket == "dislikes"
            and item.value not in allergens
        )
    ]
    likes = [
        item.value for item in effective
        if item.operation == "add" and item.bucket == "likes"
    ]
    dietary = [
        item.value for item in effective
        if item.operation == "add" and item.bucket == "dietary_constraints"
    ]
    removed = [
        item.value for item in effective if item.operation == "remove"
    ]

    facts: list[str] = []
    if lang == "en":
        if allergens:
            facts.append(f"you are allergic to {_joined(allergens, lang)}")
        if dislikes:
            facts.append(f"you don't want {_joined(dislikes, lang)}")
        if likes:
            facts.append(f"you like {_joined(likes, lang)}")
        if dietary:
            facts.append(f"{_joined(dietary, lang)} is a current goal")
        if removed:
            facts.append(f"the earlier {_joined(removed, lang)} restriction no longer applies")
        understood = "Got it: " + "; ".join(facts) + "."
        if not saved:
            return (
                understood
                + " I'll follow it in this reply, but the conversation preference could not be saved, so you may need to remind me next time."
            )
        if dietary:
            return (
                understood
                + " I'll treat it as temporary and confirm it again before using it in a later recommendation."
            )
        return understood + " I'll use that in later recipe suggestions."

    if allergens:
        facts.append(f"你对{_joined(allergens, lang)}过敏")
    if dislikes:
        facts.append(f"你不喜欢或不吃{_joined(dislikes, lang)}")
    if likes:
        facts.append(f"你喜欢{_joined(likes, lang)}")
    if dietary:
        facts.append(f"{_joined(dietary, lang)}是你现阶段的目标")
    if removed:
        facts.append(f"之前关于{_joined(removed, lang)}的限制已经取消")
    understood = "明白了：" + "；".join(facts) + "。"
    if not saved:
        return (
            understood
            + "本轮我会照这个处理，但会话偏好没有保存成功，下次可能还需要你再提醒我。"
        )
    if dietary:
        return (
            understood
            + "我会把它当成阶段性信息，之后真正用于推荐时再向你确认。"
        )
    return understood + "后续推荐我会按这个参考。"


def _continues_pending_dietary_request(
    snapshot: ShortTermRuntimeSnapshot | None,
    mutations: list[PreferenceMutation],
) -> bool:
    """撤销正在确认的阶段性目标后，继续原菜谱任务而不是只回确认文案。"""
    if snapshot is None:
        return False
    pending = snapshot.memory.task_state.pending_search_clarification
    if not isinstance(pending, dict) or str(pending.get("dimension") or "") != (
        "remembered_dietary_constraint"
    ):
        return False
    request = pending.get("request")
    if not isinstance(request, dict):
        return False
    note = request.get("context_note")
    remembered_term = str(
        (note if isinstance(note, dict) else {}).get("term") or ""
    ).strip().lower()
    if not remembered_term:
        return False
    return any(
        mutation.operation == "remove"
        and mutation.bucket == "dietary_constraints"
        and mutation.value.strip().lower() == remembered_term
        for mutation in mutations
    )


async def handle_preference_memory_turn(
    text: str,
    thread_id: str,
    *,
    short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
) -> PreferenceTurnReply | None:
    """处理显式偏好读写；包含选菜任务时保存后继续正常路由。"""
    if not supports_account_profile(thread_id):
        return None
    lang = detect_lang(text)
    service = get_conversation_service()

    if is_preference_memory_query(text):
        success = True
        try:
            runtime_memory = await service.load_runtime_memory(
                thread_id,
                current_message=text,
                short_term_snapshot=short_term_snapshot,
            )
            context = await service.routing_context(
                thread_id,
                scope="full",
                current_message=text,
                runtime_memory=runtime_memory,
            )
            if context.get("long_term_memory_status") in {
                "unavailable",
                "invalid",
            }:
                success = False
                message = (
                    "I can't read your saved food preferences right now. Please try again in a moment."
                    if lang == "en"
                    else "我现在没能读取到已保存的饮食偏好，不能把它当成没有记录；过一会儿再问我一次。"
                )
            else:
                message = _preference_summary(context, lang)
        except Exception:
            success = False
            message = (
                "I can't read your saved food preferences right now. Please try again in a moment."
                if lang == "en"
                else "我现在没能读取到你的饮食偏好，过一会儿再问我一次。"
            )
        return PreferenceTurnReply(
            message=message,
            lang=lang,
            status="read" if success else "read_failed",
            success=success,
        )

    mutations = extract_preference_mutations(text)
    if not mutations:
        return None
    channel, user_id = conversation_identity(thread_id)
    try:
        _memory, changed = await service.apply_session_preference_mutations(
            thread_id,
            mutations,
            channel=channel,
            user_id=user_id,
        )
        saved = True
    except Exception:
        changed = []
        saved = False
    changed_buckets = tuple(dict.fromkeys(item.bucket for item in changed))
    if not saved:
        status = "write_failed"
    elif changed:
        status = "saved"
    else:
        status = "unchanged"
    return PreferenceTurnReply(
        message=_mutation_ack(
            mutations,
            changed=changed,
            lang=lang,
            saved=saved,
        ),
        lang=lang,
        continue_routing=(
            has_additional_food_task(text)
            or _continues_pending_dietary_request(
                short_term_snapshot,
                mutations,
            )
        ),
        status=status,
        success=saved,
        state_changes=(
            (
                MemoryStateChange(
                    scope="session",
                    operation="set",
                    keys=changed_buckets,
                    reason_code="PREFERENCE_SESSION_UPDATED",
                ),
            )
            if changed_buckets
            else ()
        ),
    )
