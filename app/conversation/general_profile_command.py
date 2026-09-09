"""通用长期画像事实的显式“记住/忘记”命令。"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from app.conversation.command_result import MemoryStateChange
from app.conversation.profile_facts import (
    canonical_profile_fact_text,
    contains_sensitive_profile_data,
    is_explicit_profile_memory_delete,
    is_explicit_profile_memory_write,
)
from app.conversation.profile_rollout import general_profile_memory_enabled
from app.conversation.service import get_conversation_service, supports_account_profile


GeneralProfileAction = Literal["write", "delete"]
_RESET_COMMANDS = {
    "重新开始", "清空会话", "忘掉刚才", "新对话", "给我一个新对话",
    "start over", "clear chat", "forget this conversation", "new conversation",
    "清除记忆", "清空我的资料", "删除我的所有资料", "忘掉我的所有资料",
    "忘掉关于我的所有信息", "clear memory", "delete all my profile data",
    "forget everything about me",
}


@dataclass(frozen=True)
class GeneralProfileTurnReply:
    action: GeneralProfileAction
    status: str
    success: bool
    lang: str
    message: str = ""
    fact_texts: tuple[str, ...] = ()
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    changed_count: int = 0
    state_changes: tuple[MemoryStateChange, ...] = ()
    error_code: str | None = None

    def with_message(self, message: str) -> "GeneralProfileTurnReply":
        return replace(self, message=str(message or "").strip())


def _detect_lang(text: str) -> str:
    value = str(text or "")
    if re.search(r"[\u4e00-\u9fff]", value):
        return "zh"
    return "en" if re.search(r"[A-Za-z]", value) else "zh"


async def handle_general_profile_memory_turn(
    text: str,
    thread_id: str,
) -> GeneralProfileTurnReply | None:
    """只处理明确写删命令；资料查询继续交给自然回复链。"""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower().strip(
        "，。！？,.!?"
    )
    if normalized in _RESET_COMMANDS:
        return None
    explicit_write = is_explicit_profile_memory_write(text)
    explicit_delete = is_explicit_profile_memory_delete(text)
    if not explicit_write and not explicit_delete:
        return None
    if not supports_account_profile(thread_id):
        return None

    action: GeneralProfileAction = (
        "delete" if explicit_delete and not explicit_write else "write"
    )
    lang = _detect_lang(text)
    if not general_profile_memory_enabled(thread_id):
        return GeneralProfileTurnReply(
            action=action,
            status="disabled",
            success=False,
            lang=lang,
            error_code="GENERAL_PROFILE_MEMORY_DISABLED",
        )
    if contains_sensitive_profile_data(text):
        return GeneralProfileTurnReply(
            action=action,
            status="sensitive_rejected",
            success=False,
            lang=lang,
            error_code="GENERAL_PROFILE_SENSITIVE_REJECTED",
        )

    service = get_conversation_service()
    if getattr(service, "profile_fact_extractor", None) is None:
        return GeneralProfileTurnReply(
            action=action,
            status="disabled",
            success=False,
            lang=lang,
            error_code="GENERAL_PROFILE_EXTRACTOR_UNAVAILABLE",
        )
    try:
        extraction, applied = await service.capture_general_profile_facts(
            thread_id,
            text,
            force=True,
            mark_handled=True,
        )
    except Exception:
        return GeneralProfileTurnReply(
            action=action,
            status="write_failed",
            success=False,
            lang=lang,
            error_code="GENERAL_PROFILE_WRITE_FAILED",
        )
    if extraction is None:
        return GeneralProfileTurnReply(
            action=action,
            status="disabled",
            success=False,
            lang=lang,
            error_code="GENERAL_PROFILE_MEMORY_DISABLED",
        )
    if extraction.status != "success":
        return GeneralProfileTurnReply(
            action=action,
            status="extractor_failed",
            success=False,
            lang=lang,
            candidate_count=extraction.candidate_count,
            accepted_count=len(extraction.facts),
            rejected_count=extraction.rejected_count,
            error_code=extraction.error_code or "GENERAL_PROFILE_EXTRACTOR_FAILED",
        )
    if not extraction.facts:
        sensitive = contains_sensitive_profile_data(text)
        return GeneralProfileTurnReply(
            action=action,
            status="sensitive_rejected" if sensitive else "no_supported_fact",
            success=False,
            lang=lang,
            candidate_count=extraction.candidate_count,
            accepted_count=0,
            rejected_count=extraction.rejected_count,
            error_code=(
                "GENERAL_PROFILE_SENSITIVE_REJECTED"
                if sensitive
                else "GENERAL_PROFILE_NO_SUPPORTED_FACT"
            ),
        )

    if applied is None:
        return GeneralProfileTurnReply(
            action=action,
            status="write_failed",
            success=False,
            lang=lang,
            candidate_count=extraction.candidate_count,
            accepted_count=len(extraction.facts),
            rejected_count=extraction.rejected_count,
            error_code="GENERAL_PROFILE_WRITE_RESULT_MISSING",
        )
    changed_count = len(applied.upserted) + len(applied.deleted_ids)
    only_deletes = all(item.operation == "delete" for item in extraction.facts)
    if applied.changed:
        status = "deleted" if only_deletes else "saved"
    else:
        status = "not_found" if only_deletes else "unchanged"
    operation = "clear" if only_deletes else "set"
    return GeneralProfileTurnReply(
        action="delete" if only_deletes else "write",
        status=status,
        success=True,
        lang=lang,
        fact_texts=tuple(
            canonical_profile_fact_text(item, lang=lang)
            for item in extraction.facts
            if item.value
        ),
        candidate_count=extraction.candidate_count,
        accepted_count=len(extraction.facts),
        rejected_count=extraction.rejected_count,
        changed_count=changed_count,
        state_changes=(
            MemoryStateChange(
                scope="profile",
                operation=operation,
                keys=("profile_facts",),
                reason_code=(
                    "GENERAL_PROFILE_FACTS_DELETED"
                    if only_deletes
                    else "GENERAL_PROFILE_FACTS_SAVED"
                ),
            ),
        ) if applied.changed else (),
    )
