"""通用用户画像事实的受控模型与确定性校验。

LLM 只能提出候选；只有能在当前用户原文中找到证据、命中字段白名单且不含
敏感信息的候选，才允许进入长期画像。称呼、饮食偏好、过敏和设备状态继续由
现有确定性模块负责，避免同一事实出现两套写入语义。
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ProfileFactCategory = Literal[
    "identity",
    "work",
    "household",
    "habit",
    "goal",
    "cooking_profile",
    "communication",
    "interest",
]
ProfileFactKey = Literal[
    "self_description",
    "occupation",
    "industry",
    "work_focus",
    "household_member",
    "household_size",
    "relationship",
    "routine",
    "schedule",
    "cooking_habit",
    "meal_habit",
    "long_term_goal",
    "temporary_goal",
    "skill_level",
    "usual_diners",
    "time_budget",
    "cooking_frequency",
    "response_style",
    "language_preference",
    "detail_preference",
    "interest",
]
ProfileFactOperation = Literal["upsert", "delete"]
ProfileFactScope = Literal["stable", "temporal"]
ProfileFactSubject = Literal["self", "household"]
ProfileFactSensitivity = Literal["normal", "restricted", "secret"]


_ALLOWED_KEYS: dict[str, set[str]] = {
    "identity": {"self_description"},
    "work": {"occupation", "industry", "work_focus"},
    "household": {"household_member", "household_size", "relationship"},
    "habit": {"routine", "schedule", "cooking_habit", "meal_habit"},
    "goal": {"long_term_goal", "temporary_goal"},
    "cooking_profile": {
        "skill_level",
        "usual_diners",
        "time_budget",
        "cooking_frequency",
    },
    "communication": {
        "response_style",
        "language_preference",
        "detail_preference",
    },
    "interest": {"interest"},
}

_MEMORY_HINT_RE = re.compile(
    r"记住|记一下|记下来|别忘了|忘掉|忘记|别记|"
    r"(?:我|本人)(?:是|在做|从事|负责|有|平时|通常|一般|习惯|希望|想让你)|"
    r"我的(?:工作|职业|行业|家人|家庭|习惯|目标|兴趣|厨艺|作息)|"
    r"\b(?:remember|forget|i am|i'm|i work|i usually|my job|my family|my goal)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_WRITE_RE = re.compile(
    r"记住|记一下|记下来|给我记着|别忘了|"
    r"\bremember\s+(?:that|this|me)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_DELETE_RE = re.compile(
    r"忘掉|忘记|别记|不要记|删除.{0,8}(?:记忆|资料|信息)|清除.{0,8}(?:记忆|资料|信息)|"
    r"\bforget\s+(?:that|this|about me|my)\b",
    flags=re.IGNORECASE,
)
_MEMORY_QUERY_RE = re.compile(
    r"还?记得我是谁|记得我(?:什么|哪些|多少)|了解我(?:什么|多少)|"
    r"你对我知道多少|我(?:是做什么|做什么工作)|我的职业是什么|"
    r"\b(?:what do you remember about me|do you remember me|who am i|what is my job)\b",
    flags=re.IGNORECASE,
)
_SELF_EVIDENCE_RE = re.compile(
    r"(?:我|本人|我的)|\b(?:i|i'm|i am|my)\b",
    flags=re.IGNORECASE,
)
_HOUSEHOLD_EVIDENCE_RE = re.compile(
    r"我家|家里人|家人|家庭|老婆|妻子|太太|老公|丈夫|孩子|女儿|儿子|父母|"
    r"\bmy\s+(?:family|wife|husband|child|children|daughter|son|parents?)\b",
    flags=re.IGNORECASE,
)
_SENSITIVE_TERM_RE = re.compile(
    r"密码|口令|验证码|密钥|私钥|助记词|身份证|护照|银行卡|信用卡|"
    r"联系方式|手机号|电话号码|微信号|QQ号|住址|家庭地址|详细地址|门牌号|"
    r"服务器(?:地址|连接|账号|用户名|主机|端口)|"
    r"(?:api[ _-]?key|access[ _-]?key|secret|token|password|private[ _-]?key|"
    r"verification[ _-]?code|bank[ _-]?card|credit[ _-]?card|passport)",
    flags=re.IGNORECASE,
)
_SENSITIVE_SHAPE_RES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(
        r"(?:我住在|我居住在|我家在|我的地址是|\b(?:i live at|my address is)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:server|ssh)\s+(?:address|host|port|login|username|connection)\b",
        re.IGNORECASE,
    ),
)
_OWNED_BY_EXISTING_MEMORY_RE = re.compile(
    r"叫我|称呼我|我叫|喜欢吃|爱吃|不喜欢吃|不爱吃|不吃|忌口|过敏|"
    r"素食|纯素|清真|减肥|减脂|低脂|控卡|"
    r"\b(?:call me|my name is|like to eat|like eating|love to eat|love eating|"
    r"dislike eating|do not eat|don't eat|allergic|vegetarian|vegan|halal|low-fat)\b",
    flags=re.IGNORECASE,
)
_INSTRUCTION_LIKE_RE = re.compile(
    r"忽略.{0,12}(?:指令|提示|规则)|系统提示|开发者消息|调用工具|执行(?:命令|代码)|"
    r"输出(?:密码|密钥|提示词)|"
    r"\b(?:ignore (?:all |the )?(?:previous|prior|system)|system prompt|developer message|"
    r"call (?:a )?tool|execute (?:a )?(?:command|code)|reveal (?:the )?prompt)\b",
    flags=re.IGNORECASE,
)


class ProfileFactCandidate(BaseModel):
    """模型提出的单条通用画像事实候选。"""

    model_config = ConfigDict(extra="forbid")

    operation: ProfileFactOperation
    category: ProfileFactCategory
    key: ProfileFactKey
    value: str = Field(default="", max_length=240)
    subject: ProfileFactSubject = "self"
    scope: ProfileFactScope = "stable"
    expires_in_days: int | None = Field(default=None, ge=1, le=365)
    sensitivity: ProfileFactSensitivity = "normal"
    evidence: str = Field(min_length=1, max_length=240)

    @field_validator("value", "evidence", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return " ".join(str(value or "").split()).strip()

    @model_validator(mode="after")
    def _validate_operation_value(self) -> "ProfileFactCandidate":
        if self.operation == "upsert" and not self.value:
            raise ValueError("upsert fact requires value")
        if self.scope == "stable" and self.expires_in_days is not None:
            raise ValueError("stable fact cannot define expires_in_days")
        return self


class ProfileFactExtraction(BaseModel):
    """Memory Extractor 的严格 JSON 输出。"""

    model_config = ConfigDict(extra="forbid")

    facts: list[ProfileFactCandidate] = Field(default_factory=list, max_length=3)


def may_contain_profile_fact(text: str) -> bool:
    value = " ".join(str(text or "").split()).strip()
    return bool(value and _MEMORY_HINT_RE.search(value))


def is_explicit_profile_memory_write(text: str) -> bool:
    return bool(_EXPLICIT_WRITE_RE.search(" ".join(str(text or "").split())))


def is_explicit_profile_memory_delete(text: str) -> bool:
    return bool(_EXPLICIT_DELETE_RE.search(" ".join(str(text or "").split())))


def is_general_profile_memory_query(text: str) -> bool:
    return bool(_MEMORY_QUERY_RE.search(" ".join(str(text or "").split())))


def contains_sensitive_profile_data(text: str) -> bool:
    value = str(text or "")
    return bool(
        _SENSITIVE_TERM_RE.search(value)
        or any(pattern.search(value) for pattern in _SENSITIVE_SHAPE_RES)
    )


def _normalized_evidence(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _candidate_is_grounded(
    message: str,
    candidate: ProfileFactCandidate,
    *,
    explicit_request: bool,
) -> bool:
    if candidate.sensitivity != "normal":
        return False
    if candidate.key not in _ALLOWED_KEYS.get(candidate.category, set()):
        return False
    if contains_sensitive_profile_data(candidate.evidence) or contains_sensitive_profile_data(
        candidate.value
    ):
        return False
    if _INSTRUCTION_LIKE_RE.search(candidate.evidence) or _INSTRUCTION_LIKE_RE.search(
        candidate.value
    ):
        return False
    normalized_message = _normalized_evidence(message)
    normalized_evidence = _normalized_evidence(candidate.evidence)
    if not normalized_evidence or normalized_evidence not in normalized_message:
        return False
    if candidate.operation == "upsert":
        normalized_value = _normalized_evidence(candidate.value)
        if not normalized_value or normalized_value not in normalized_evidence:
            return False
    if not explicit_request:
        marker = (
            _HOUSEHOLD_EVIDENCE_RE
            if candidate.subject == "household"
            else _SELF_EVIDENCE_RE
        )
        if not marker.search(candidate.evidence):
            return False
    # 这些事实已有更严格的确定性写入器，通用 Extractor 不得重复拥有。
    if _OWNED_BY_EXISTING_MEMORY_RE.search(candidate.evidence):
        return False
    return True


def filter_grounded_profile_facts(
    message: str,
    candidates: list[ProfileFactCandidate],
) -> list[ProfileFactCandidate]:
    """返回可安全写入的候选；不记录或返回被拒候选的正文。"""
    explicit_request = bool(
        is_explicit_profile_memory_write(message)
        or is_explicit_profile_memory_delete(message)
    )
    accepted: list[ProfileFactCandidate] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for candidate in candidates[:3]:
        if not _candidate_is_grounded(
            message,
            candidate,
            explicit_request=explicit_request,
        ):
            continue
        normalized = candidate
        if candidate.scope == "temporal" and candidate.expires_in_days is None:
            normalized = candidate.model_copy(update={"expires_in_days": 30})
        signature = (
            normalized.operation,
            normalized.category,
            normalized.key,
            normalized.subject,
            normalized.value.lower(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        accepted.append(normalized)
    return accepted


_FACT_LABELS = {
    "self_description": "自我描述",
    "occupation": "职业",
    "industry": "行业",
    "work_focus": "工作方向",
    "household_member": "家庭成员",
    "household_size": "家庭人数",
    "relationship": "家庭关系",
    "routine": "日常习惯",
    "schedule": "作息",
    "cooking_habit": "做饭习惯",
    "meal_habit": "用餐习惯",
    "long_term_goal": "长期目标",
    "temporary_goal": "阶段目标",
    "skill_level": "厨艺水平",
    "usual_diners": "通常用餐人数",
    "time_budget": "做饭时间预算",
    "cooking_frequency": "做饭频率",
    "response_style": "回复风格",
    "language_preference": "沟通语言",
    "detail_preference": "回答详略",
    "interest": "兴趣",
}
_FACT_LABELS_EN = {
    "self_description": "self-description",
    "occupation": "occupation",
    "industry": "industry",
    "work_focus": "work focus",
    "household_member": "household member",
    "household_size": "household size",
    "relationship": "relationship",
    "routine": "routine",
    "schedule": "schedule",
    "cooking_habit": "cooking habit",
    "meal_habit": "meal habit",
    "long_term_goal": "long-term goal",
    "temporary_goal": "temporary goal",
    "skill_level": "cooking skill level",
    "usual_diners": "usual diners",
    "time_budget": "cooking time budget",
    "cooking_frequency": "cooking frequency",
    "response_style": "preferred response style",
    "language_preference": "language preference",
    "detail_preference": "detail preference",
    "interest": "interest",
}

_MULTI_VALUE_KEYS = {
    "household_member",
    "relationship",
    "work_focus",
    "long_term_goal",
    "temporary_goal",
    "interest",
}
GENERAL_PROFILE_FACT_TYPE = "profile_fact"


def canonical_profile_fact_text(
    candidate: ProfileFactCandidate,
    *,
    lang: str = "zh",
) -> str:
    """构造不含用户 ID 或整句原文的候选事实文本。"""
    if lang == "en":
        label = _FACT_LABELS_EN.get(candidate.key, candidate.key)
        subject = "User household" if candidate.subject == "household" else "User"
        return f"{subject} {label}: {candidate.value}"[:512]
    label = _FACT_LABELS.get(candidate.key, candidate.key)
    subject = "用户家庭" if candidate.subject == "household" else "用户"
    return f"{subject}{label}：{candidate.value}"[:512]


def canonical_stored_profile_fact_text(fact: dict, *, lang: str = "zh") -> str:
    """为已持久化事实生成提示词/向量文本，不读取来源线程。"""
    key = str(fact.get("key") or "")
    value = str(fact.get("value") or "").strip()
    if lang == "en":
        label = _FACT_LABELS_EN.get(key, key)
        subject = "User household" if fact.get("subject") == "household" else "User"
        return f"{subject} {label}: {value}"[:512]
    label = _FACT_LABELS.get(key, key)
    subject = "用户家庭" if fact.get("subject") == "household" else "用户"
    return f"{subject}{label}：{value}"[:512]


def _fact_slot(value: ProfileFactCandidate | dict) -> tuple[str, str, str]:
    if isinstance(value, ProfileFactCandidate):
        return value.category, value.key, value.subject
    return (
        str(value.get("category") or ""),
        str(value.get("key") or ""),
        str(value.get("subject") or "self"),
    )


def _normalized_fact_value(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _value_matches(stored: object, requested: object) -> bool:
    left = _normalized_fact_value(stored)
    right = _normalized_fact_value(requested)
    if not right:
        return True
    if not left:
        return False
    return left == right or (
        min(len(left), len(right)) >= 2 and (left in right or right in left)
    )


def _profile_fact_id(profile_key: str, candidate: ProfileFactCandidate) -> str:
    fingerprint = hashlib.sha256(
        "\0".join((
            str(profile_key or ""),
            candidate.category,
            candidate.key,
            candidate.subject,
            _normalized_fact_value(candidate.value),
        )).encode("utf-8")
    ).hexdigest()
    return f"mem_{fingerprint[:24]}"


def is_active_general_profile_fact(fact: dict, *, now: float | None = None) -> bool:
    current = time.time() if now is None else float(now)
    if not isinstance(fact, dict):
        return False
    if str(fact.get("type") or "") != GENERAL_PROFILE_FACT_TYPE:
        return False
    if str(fact.get("status") or "active") != "active":
        return False
    if not str(fact.get("value") or "").strip():
        return False
    expires_at = float(fact.get("expires_at") or 0)
    return not expires_at or expires_at > current


def active_general_profile_facts(
    facts: list[dict],
    *,
    now: float | None = None,
    limit: int = 50,
) -> list[dict]:
    current = time.time() if now is None else float(now)
    active = [
        dict(item)
        for item in (facts or [])
        if is_active_general_profile_fact(item, now=current)
    ]
    active.sort(
        key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0),
        reverse=True,
    )
    return active[:max(1, int(limit))]


@dataclass(frozen=True)
class ProfileFactApplyResult:
    facts: list[dict]
    changed: bool
    upserted: tuple[dict, ...] = ()
    deleted_ids: tuple[str, ...] = ()


def apply_profile_fact_mutations(
    facts: list[dict],
    candidates: list[ProfileFactCandidate],
    *,
    profile_key: str,
    source_channel: str,
    source_thread_id: str,
    now: float | None = None,
) -> ProfileFactApplyResult:
    """将已校验候选应用到通用事实；用户删除和过期值会物理移除。"""
    current = time.time() if now is None else float(now)
    source_thread_hash = hashlib.sha256(
        str(source_thread_id or "").encode("utf-8")
    ).hexdigest()[:24]
    original = [dict(item) for item in (facts or []) if isinstance(item, dict)]
    # 通用事实采用可删除的当前态，不保留已过期或 inactive 的值；旧称呼/偏好
    # 审计事实仍维持原实现，不在这里处理。
    result = [
        item
        for item in original
        if (
            str(item.get("type") or "") != GENERAL_PROFILE_FACT_TYPE
            or is_active_general_profile_fact(item, now=current)
        )
    ]
    upserted: list[dict] = []
    deleted_ids: list[str] = [
        str(item.get("id") or "")
        for item in original
        if (
            str(item.get("type") or "") == GENERAL_PROFILE_FACT_TYPE
            and not is_active_general_profile_fact(item, now=current)
            and str(item.get("id") or "")
        )
    ]

    for candidate in candidates[:3]:
        slot = _fact_slot(candidate)
        matching = [
            item
            for item in result
            if (
                str(item.get("type") or "") == GENERAL_PROFILE_FACT_TYPE
                and _fact_slot(item) == slot
            )
        ]
        if candidate.operation == "delete":
            targets = [
                item
                for item in matching
                if _value_matches(item.get("value"), candidate.value)
            ]
            target_ids = {str(item.get("id") or "") for item in targets}
            result = [
                item for item in result if str(item.get("id") or "") not in target_ids
            ]
            deleted_ids.extend(item for item in target_ids if item)
            continue

        if candidate.key not in _MULTI_VALUE_KEYS:
            obsolete = [
                item
                for item in matching
                if not _value_matches(item.get("value"), candidate.value)
            ]
            obsolete_ids = {str(item.get("id") or "") for item in obsolete}
            result = [
                item for item in result if str(item.get("id") or "") not in obsolete_ids
            ]
            deleted_ids.extend(item for item in obsolete_ids if item)
            matching = [item for item in matching if item not in obsolete]

        existing = next(
            (
                item
                for item in matching
                if _value_matches(item.get("value"), candidate.value)
            ),
            None,
        )
        expires_at = (
            current + int(candidate.expires_in_days or 30) * 86_400
            if candidate.scope == "temporal"
            else None
        )
        record = {
            "id": (
                str(existing.get("id") or "")
                if existing is not None
                else _profile_fact_id(profile_key, candidate)
            ),
            "type": GENERAL_PROFILE_FACT_TYPE,
            "category": candidate.category,
            "key": candidate.key,
            "value": candidate.value,
            "subject": candidate.subject,
            "scope": candidate.scope,
            "source": "user_explicit",
            "source_channel": str(source_channel or "unknown"),
            "source_thread_hash": source_thread_hash,
            "confidence": 1.0,
            "status": "active",
            "created_at": (
                float(existing.get("created_at") or current)
                if existing is not None
                else current
            ),
            "updated_at": current,
        }
        if expires_at is not None:
            record["expires_at"] = expires_at
        if existing is not None:
            index = result.index(existing)
            result[index] = record
        else:
            result.append(record)
        upserted.append(dict(record))

    # PostgreSQL Store 自身还会做总上限保护；这里优先保留旧审计事实和最近通用事实。
    non_general = [
        item for item in result if str(item.get("type") or "") != GENERAL_PROFILE_FACT_TYPE
    ]
    general = [
        item for item in result if str(item.get("type") or "") == GENERAL_PROFILE_FACT_TYPE
    ][-50:]
    final = [*non_general, *general][-100:]
    changed = final != original
    return ProfileFactApplyResult(
        facts=final,
        changed=changed,
        upserted=tuple(upserted),
        deleted_ids=tuple(dict.fromkeys(deleted_ids)),
    )
