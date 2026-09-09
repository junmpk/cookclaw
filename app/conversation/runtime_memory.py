"""单轮运行时记忆快照。

该模块只描述一次业务轮次已经加载的数据，不拥有任何持久化语义：

- ``short_term`` 是 Redis ``ConversationMemory`` 的隔离副本；
- ``long_term`` 是 PostgreSQL 账号画像的当前有效只读投影；
- 快照不会写回 Redis，也不能作为设备动作授权。

长期资料只在业务层按用途裁剪后进入 Router、Planner 或回复模型。完整快照本身
不得写入 Trace、日志或通道响应。
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from app.conversation.models import ConversationMemory

ShortTermLoadStatus = Literal[
    "loaded",
    "new",
    "unavailable",
    "invalid",
    "unsupported",
]
TaskStateLoadStatus = Literal[
    "loaded",
    "legacy",
    "new",
    "unavailable",
    "invalid",
    "unsupported",
]
LongTermLoadStatus = Literal[
    "loaded",
    "empty",
    "unavailable",
    "invalid",
    "unsupported",
]
MemoryScope = Literal["preferences", "route", "full"]

PreferenceBucketName = Literal[
    "likes",
    "dislikes",
    "allergens",
    "dietary_constraints",
    "available_ingredients",
]
PreferenceMutationOperation = Literal["add", "remove"]
PreferenceMutationScope = Literal[
    "session",
    "temporal",
    "long_term_candidate",
]


@dataclass(frozen=True)
class ShortTermRuntimeSnapshot:
    """Redis 短期会话的一次只读加载结果，可被 task scope 与整轮快照复用。"""

    thread_id: str = field(repr=False)
    channel: str
    user_id: str | None = field(repr=False)
    memory: ConversationMemory = field(repr=False)
    status: ShortTermLoadStatus = "new"
    is_new_session: bool | None = True
    task_state_status: TaskStateLoadStatus = "new"
    task_state_revision: int = 0
    task_state_generation: int = 0

    def belongs_to(self, thread_id: str) -> bool:
        return self.thread_id == str(thread_id or "")


@dataclass(frozen=True)
class LongTermMemorySnapshot:
    """账号长期画像的单轮只读投影；PostgreSQL 仍是唯一事实源。"""

    status: LongTermLoadStatus = "empty"
    profile_version: int = 0
    preferred_name: str | None = field(default=None, repr=False)
    display_name: str | None = field(default=None, repr=False)
    preferences: dict[str, list[str]] = field(default_factory=dict, repr=False)
    temporal_dietary_constraints: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    account_digest: list[str] = field(default_factory=list, repr=False)
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    # 画像中的删除/过期墓碑只用于阻止旧 Redis thread 复活已撤销偏好，
    # 不进入任何 Prompt 或通道响应。
    invalidated_preferences: dict[str, list[str]] = field(
        default_factory=dict,
        repr=False,
    )
    # 当前有效长期事实的内部投影，包含饮食事实和通用画像事实。调用方只能
    # 使用受控视图，不能把整个列表直接拼入 Prompt。
    facts: list[dict[str, Any]] = field(default_factory=list, repr=False)
    general_facts: list[dict[str, Any]] = field(default_factory=list, repr=False)
    food_memory_cleared_at: float = 0.0
    loaded_at: float = field(default_factory=time.time)

    @property
    def available(self) -> bool:
        return self.status in {"loaded", "empty"}

    @property
    def has_data(self) -> bool:
        return bool(
            self.preferred_name
            or self.display_name
            or any(self.preferences.values())
            or self.temporal_dietary_constraints
            or self.account_digest
            or self.events
            or self.general_facts
        )

    def context_ref(self) -> dict[str, Any]:
        """只返回可观测形状；不包含称呼、偏好、事实值或用户身份。"""
        return {
            "status": self.status,
            "profile_version": max(0, int(self.profile_version or 0)),
            "has_preferred_name": bool(self.preferred_name),
            "preference_count": sum(
                len(values) for values in self.preferences.values()
            ),
            "temporal_constraint_count": len(
                self.temporal_dietary_constraints
            ),
            "event_count": len(self.events),
            "general_fact_count": len(self.general_facts),
        }

    def copy_general_facts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return deepcopy(self.general_facts[:max(0, int(limit))])


@dataclass(frozen=True, slots=True)
class MemoryViewLoadState:
    """类型化视图的加载语义。

    ``provided=False`` 表示调用方根本没有提供本轮快照；它与“已经读取、但画像
    为空”的 ``provided=True + long_term_status='empty'`` 明确不同。后端不可用
    或数据损坏也保留各自状态，消费者不能再通过容器真假值猜测加载结果。
    """

    provided: bool
    short_term_status: ShortTermLoadStatus | None = None
    long_term_status: LongTermLoadStatus | None = None
    profile_version: int = 0
    is_new_session: bool | None = None
    hydrated_at: float = 0.0

    @property
    def short_term_available(self) -> bool:
        return self.provided and self.short_term_status in {"loaded", "new"}

    @property
    def long_term_available(self) -> bool:
        return self.provided and self.long_term_status in {"loaded", "empty"}

    @property
    def long_term_loaded_empty(self) -> bool:
        return self.provided and self.long_term_status == "empty"

    @property
    def long_term_failed(self) -> bool:
        return self.provided and self.long_term_status in {
            "unavailable",
            "invalid",
        }


@dataclass(frozen=True, slots=True, repr=False)
class PreferenceValuesView:
    """已复制成元组的饮食偏好；不会暴露来源字典或列表。"""

    likes: tuple[str, ...] = ()
    dislikes: tuple[str, ...] = ()
    allergens: tuple[str, ...] = ()
    dietary_constraints: tuple[str, ...] = ()
    available_ingredients: tuple[str, ...] = ()

    def values_for(self, bucket: PreferenceBucketName) -> tuple[str, ...]:
        return getattr(self, bucket)

    @property
    def has_values(self) -> bool:
        return any((
            self.likes,
            self.dislikes,
            self.allergens,
            self.dietary_constraints,
            self.available_ingredients,
        ))

    def to_context_dict(self) -> dict[str, list[str]]:
        """返回隔离的兼容字典，不暴露视图中的元组。"""
        return _preference_context(self)


@dataclass(frozen=True, slots=True, repr=False)
class SafetyConstraintsView:
    """只包含安全路由需要的负向或强约束，不携带喜好和库存。"""

    dislikes: tuple[str, ...] = ()
    allergens: tuple[str, ...] = ()
    dietary_constraints: tuple[str, ...] = ()

    @property
    def has_values(self) -> bool:
        return any((self.dislikes, self.allergens, self.dietary_constraints))


@dataclass(frozen=True, slots=True, repr=False)
class PreferenceMutationView:
    """当前轮偏好变更的不可变最小投影；不携带原始 source span。"""

    operation: PreferenceMutationOperation
    bucket: PreferenceBucketName
    value: str
    scope: PreferenceMutationScope


@dataclass(frozen=True, slots=True, repr=False)
class TemporalDietaryConstraintView:
    """阶段性饮食约束的只读投影。"""

    value: str
    source: str = ""
    updated_at: float = 0.0
    expires_at: float = 0.0


@dataclass(frozen=True, slots=True, repr=False)
class GeneralProfileFactView:
    """通用账号事实的允许字段投影，不保留任意元数据字典。"""

    fact_id: str
    category: str
    key: str
    value: str
    subject: str = "self"
    scope: str = "stable"
    source: str = ""
    updated_at: float = 0.0
    expires_at: float = 0.0


@dataclass(frozen=True, slots=True, repr=False)
class ConversationTurnView:
    """近期一轮消息的隔离副本。"""

    role: str
    content: str
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class SafetyMemoryView:
    """过敏与饮食安全门禁所需的最小、分来源视图。

    这里故意不合并来源。长期、会话和当前轮的覆盖策略仍应由确定性业务规则
    决定；视图只保证消费者无法修改 Runtime 快照，也不会误把加载失败当空画像。
    """

    load_state: MemoryViewLoadState
    long_term_constraints: SafetyConstraintsView = field(
        default_factory=SafetyConstraintsView,
        repr=False,
    )
    session_constraints: SafetyConstraintsView = field(
        default_factory=SafetyConstraintsView,
        repr=False,
    )
    current_turn_constraints: SafetyConstraintsView = field(
        default_factory=SafetyConstraintsView,
        repr=False,
    )
    invalidated_long_term_constraints: SafetyConstraintsView = field(
        default_factory=SafetyConstraintsView,
        repr=False,
    )
    temporal_dietary_constraints: tuple[
        TemporalDietaryConstraintView,
        ...,
    ] = field(default_factory=tuple, repr=False)
    current_turn_mutations: tuple[PreferenceMutationView, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    # 由 ConversationService 按墓碑、clear marker 和本轮纠正规则解析。
    # 原始 snapshot builder 不擅自合并来源，因此默认保持 unresolved。
    policy_resolved: bool = False
    effective_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    effective_temporal_dietary_constraints: tuple[
        TemporalDietaryConstraintView,
        ...,
    ] = field(default_factory=tuple, repr=False)

    def to_context_dict(self) -> dict[str, Any]:
        """返回一次性兼容字典；修改返回值不会污染视图或 Runtime 快照。"""
        return {
            **_load_state_context(self.load_state),
            "long_term_constraints": _safety_context(
                self.long_term_constraints
            ),
            "session_constraints": _safety_context(self.session_constraints),
            "current_turn_constraints": _safety_context(
                self.current_turn_constraints
            ),
            "invalidated_long_term_constraints": _safety_context(
                self.invalidated_long_term_constraints
            ),
            "temporal_dietary_constraints": [
                _temporal_context(item)
                for item in self.temporal_dietary_constraints
            ],
            "current_turn_mutations": [
                _mutation_context(item) for item in self.current_turn_mutations
            ],
        }

    def to_router_context_dict(self) -> dict[str, Any]:
        """输出 Router 安全门禁使用的已解析兼容形状。"""
        if not self.policy_resolved:
            raise RuntimeError(
                "safety memory view has not been resolved by ConversationService"
            )
        return {
            **_compat_load_state_context(self.load_state),
            "preferences": _preference_context(self.effective_preferences),
            "temporal_dietary_constraints": [
                _temporal_context(item)
                for item in self.effective_temporal_dietary_constraints
            ],
        }


@dataclass(frozen=True, slots=True)
class PlannerMemoryView:
    """Planner 可消费的短期话题和分层稳定约束。"""

    load_state: MemoryViewLoadState
    long_term_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    session_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    current_turn_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    temporal_dietary_constraints: tuple[
        TemporalDietaryConstraintView,
        ...,
    ] = field(default_factory=tuple, repr=False)
    current_turn_mutations: tuple[PreferenceMutationView, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    recent_user_turns: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    previous_candidate_count: int = 0
    # 消费者只能使用服务层解析后的 effective_* 字段做业务决策。
    policy_resolved: bool = False
    effective_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    effective_temporal_dietary_constraints: tuple[
        TemporalDietaryConstraintView,
        ...,
    ] = field(default_factory=tuple, repr=False)

    def to_context_dict(self) -> dict[str, Any]:
        """返回 Planner 迁移期可序列化的新副本。"""
        return {
            **_load_state_context(self.load_state),
            "long_term_preferences": _preference_context(
                self.long_term_preferences
            ),
            "session_preferences": _preference_context(
                self.session_preferences
            ),
            "current_turn_preferences": _preference_context(
                self.current_turn_preferences
            ),
            "temporal_dietary_constraints": [
                _temporal_context(item)
                for item in self.temporal_dietary_constraints
            ],
            "current_turn_mutations": [
                _mutation_context(item) for item in self.current_turn_mutations
            ],
            "recent_user_turns": list(self.recent_user_turns),
            "previous_candidate_count": self.previous_candidate_count,
        }

    def to_planner_context_dict(self) -> dict[str, Any]:
        """输出 Planner 的最小、已解析兼容形状。"""
        if not self.policy_resolved:
            raise RuntimeError(
                "planner memory view has not been resolved by ConversationService"
            )
        return {
            **_compat_load_state_context(self.load_state),
            "preferences": _preference_context(self.effective_preferences),
            "temporal_dietary_constraints": [
                _temporal_context(item)
                for item in self.effective_temporal_dietary_constraints
            ],
            "recent_user_turns": list(self.recent_user_turns),
            "previous_candidate_count": self.previous_candidate_count,
        }


@dataclass(frozen=True, slots=True)
class QAMemoryView:
    """问答模型可用的账号资料与有限近期上下文。

    该对象仍只是数据载体，不决定消息角色。后续 Prompt 层应把它序列化为不可信
    数据，而不是把其中的用户文本提升成 System 指令。
    """

    load_state: MemoryViewLoadState
    preferred_name: str | None = field(default=None, repr=False)
    display_name: str | None = field(default=None, repr=False)
    long_term_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    session_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    current_turn_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    temporal_dietary_constraints: tuple[
        TemporalDietaryConstraintView,
        ...,
    ] = field(default_factory=tuple, repr=False)
    current_turn_mutations: tuple[PreferenceMutationView, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    account_digest: tuple[str, ...] = field(default_factory=tuple, repr=False)
    general_facts: tuple[GeneralProfileFactView, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    conversation_digest: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    recent_turns: tuple[ConversationTurnView, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    policy_resolved: bool = False
    effective_preferences: PreferenceValuesView = field(
        default_factory=PreferenceValuesView,
        repr=False,
    )
    effective_temporal_dietary_constraints: tuple[
        TemporalDietaryConstraintView,
        ...,
    ] = field(default_factory=tuple, repr=False)

    def to_context_dict(self) -> dict[str, Any]:
        """返回 QA 序列化所需的新副本，不返回任何底层容器引用。"""
        return {
            **_load_state_context(self.load_state),
            "preferred_name": self.preferred_name,
            "display_name": self.display_name,
            "long_term_preferences": _preference_context(
                self.long_term_preferences
            ),
            "session_preferences": _preference_context(
                self.session_preferences
            ),
            "current_turn_preferences": _preference_context(
                self.current_turn_preferences
            ),
            "temporal_dietary_constraints": [
                _temporal_context(item)
                for item in self.temporal_dietary_constraints
            ],
            "current_turn_mutations": [
                _mutation_context(item) for item in self.current_turn_mutations
            ],
            "account_digest": list(self.account_digest),
            "general_facts": [
                _profile_fact_context(item) for item in self.general_facts
            ],
            "conversation_digest": list(self.conversation_digest),
            "recent_turns": [
                {
                    "role": item.role,
                    "content": item.content,
                    "created_at": item.created_at,
                }
                for item in self.recent_turns
            ],
        }

    def to_qa_context_dict(self) -> dict[str, Any]:
        """输出 QA 的已解析资料与有限近期对话副本。"""
        if not self.policy_resolved:
            raise RuntimeError(
                "QA memory view has not been resolved by ConversationService"
            )
        return {
            **_compat_load_state_context(self.load_state),
            "preferred_name": self.preferred_name,
            "display_name": self.display_name,
            "preferences": _preference_context(self.effective_preferences),
            "temporal_dietary_constraints": [
                _temporal_context(item)
                for item in self.effective_temporal_dietary_constraints
            ],
            "account_digest": list(self.account_digest),
            "conversation_summary": {
                "conversation_digest": list(self.conversation_digest),
            },
            "recent_turns": [
                {
                    "role": item.role,
                    "content": item.content,
                    "created_at": item.created_at,
                }
                for item in self.recent_turns
            ],
        }


@dataclass(frozen=True)
class RuntimeMemorySnapshot:
    """短期会话与长期画像在一次业务轮次中的一致视图。"""

    thread_id: str = field(repr=False)
    channel: str
    user_id: str | None = field(repr=False)
    short_term: ConversationMemory = field(repr=False)
    short_term_status: ShortTermLoadStatus = "new"
    long_term: LongTermMemorySnapshot = field(
        default_factory=LongTermMemorySnapshot,
        repr=False,
    )
    # 当账号画像已经清除、但旧 thread 仍在 Redis 中时，只允许本轮用户
    # 重新明确表达的值覆盖清除标记，不能继续合并旧 thread 的整份偏好。
    current_turn_preferences: dict[str, list[str]] = field(
        default_factory=dict,
        repr=False,
    )
    current_turn_preference_mutations: tuple[Any, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    # Redis 读取成功时才能确认是否是新会话；后端不可用时必须保留未知。
    is_new_session: bool | None = True
    hydrated_at: float = field(default_factory=time.time)

    def belongs_to(self, thread_id: str) -> bool:
        return self.thread_id == str(thread_id or "")

    def safety_view(self) -> SafetyMemoryView:
        return build_safety_memory_view(self)

    def planner_view(self) -> PlannerMemoryView:
        return build_planner_memory_view(self)

    def qa_view(self) -> QAMemoryView:
        return build_qa_memory_view(self)

    def context_ref(self) -> dict[str, Any]:
        """供 Trace 使用的脱敏摘要，不泄露 thread、账号或记忆正文。"""
        try:
            session_version = max(0, int(self.short_term.version or 0))
        except (TypeError, ValueError):
            session_version = 0
        long_term_ref = self.long_term.context_ref()
        return {
            "short_term_status": self.short_term_status,
            "is_new_session": self.is_new_session,
            "session_version": session_version,
            "recent_turn_count": len(self.short_term.recent_turns),
            "has_summary": bool(self.short_term.summary),
            **{
                f"long_term_{key}": value
                for key, value in long_term_ref.items()
            },
        }


def _clean_text(value: object, *, limit: int) -> str:
    return str(value or "").strip()[:max(0, int(limit))]


def _safe_float(value: object) -> float:
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _safe_nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _text_values(
    value: object,
    *,
    limit: int,
    text_limit: int,
) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values = (value,)
    elif isinstance(value, (list, tuple)):
        raw_values = value
    else:
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        item = _clean_text(raw, limit=text_limit)
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
        if len(cleaned) >= max(0, int(limit)):
            break
    return tuple(cleaned)


def _preference_values(value: object) -> PreferenceValuesView:
    source = value if isinstance(value, dict) else {}
    return PreferenceValuesView(
        likes=_text_values(source.get("likes"), limit=50, text_limit=80),
        dislikes=_text_values(
            source.get("dislikes"),
            limit=50,
            text_limit=80,
        ),
        allergens=_text_values(
            source.get("allergens"),
            limit=50,
            text_limit=80,
        ),
        dietary_constraints=_text_values(
            source.get("dietary_constraints"),
            limit=50,
            text_limit=80,
        ),
        available_ingredients=_text_values(
            source.get("available_ingredients"),
            limit=50,
            text_limit=80,
        ),
    )


def _safety_constraints(
    preferences: PreferenceValuesView,
) -> SafetyConstraintsView:
    return SafetyConstraintsView(
        dislikes=preferences.dislikes,
        allergens=preferences.allergens,
        dietary_constraints=preferences.dietary_constraints,
    )


def _temporal_constraints(
    value: object,
) -> tuple[TemporalDietaryConstraintView, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    constraints: list[TemporalDietaryConstraintView] = []
    seen: set[tuple[str, float]] = set()
    for raw in value[:50]:
        if not isinstance(raw, dict):
            continue
        constraint_value = _clean_text(raw.get("value"), limit=80)
        if not constraint_value:
            continue
        expires_at = _safe_float(raw.get("expires_at"))
        signature = (constraint_value, expires_at)
        if signature in seen:
            continue
        seen.add(signature)
        constraints.append(TemporalDietaryConstraintView(
            value=constraint_value,
            source=_clean_text(raw.get("source"), limit=40),
            updated_at=_safe_float(
                raw.get("updated_at") or raw.get("last_confirmed_at")
            ),
            expires_at=expires_at,
        ))
    return tuple(constraints)


def _preference_mutations(
    value: object,
) -> tuple[PreferenceMutationView, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    allowed_operations = {"add", "remove"}
    allowed_buckets = {
        "likes",
        "dislikes",
        "allergens",
        "dietary_constraints",
        "available_ingredients",
    }
    allowed_scopes = {"session", "temporal", "long_term_candidate"}
    mutations: list[PreferenceMutationView] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in value[:50]:
        operation = _clean_text(getattr(raw, "operation", ""), limit=16)
        bucket = _clean_text(getattr(raw, "bucket", ""), limit=32)
        item_value = _clean_text(getattr(raw, "value", ""), limit=80)
        scope = _clean_text(getattr(raw, "scope", ""), limit=32)
        if (
            operation not in allowed_operations
            or bucket not in allowed_buckets
            or not item_value
            or scope not in allowed_scopes
        ):
            continue
        signature = (operation, bucket, item_value, scope)
        if signature in seen:
            continue
        seen.add(signature)
        mutations.append(PreferenceMutationView(
            operation=operation,  # type: ignore[arg-type]
            bucket=bucket,  # type: ignore[arg-type]
            value=item_value,
            scope=scope,  # type: ignore[arg-type]
        ))
    return tuple(mutations)


def _general_profile_facts(
    value: object,
) -> tuple[GeneralProfileFactView, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    facts: list[GeneralProfileFactView] = []
    seen: set[str] = set()
    for raw in value[:50]:
        if not isinstance(raw, dict):
            continue
        fact_value = _clean_text(raw.get("value"), limit=512)
        if not fact_value:
            continue
        fact_id = _clean_text(raw.get("id"), limit=80)
        signature = fact_id or "|".join((
            _clean_text(raw.get("category"), limit=40),
            _clean_text(raw.get("key"), limit=80),
            fact_value,
        ))
        if signature in seen:
            continue
        seen.add(signature)
        facts.append(GeneralProfileFactView(
            fact_id=fact_id,
            category=_clean_text(raw.get("category"), limit=40),
            key=_clean_text(raw.get("key"), limit=80),
            value=fact_value,
            subject=_clean_text(raw.get("subject"), limit=40) or "self",
            scope=_clean_text(raw.get("scope"), limit=40) or "stable",
            source=_clean_text(raw.get("source"), limit=40),
            updated_at=_safe_float(raw.get("updated_at")),
            expires_at=_safe_float(raw.get("expires_at")),
        ))
    return tuple(facts)


def _recent_turns(memory: ConversationMemory) -> tuple[ConversationTurnView, ...]:
    turns: list[ConversationTurnView] = []
    for raw in list(memory.recent_turns or [])[-8:]:
        role = _clean_text(getattr(raw, "role", ""), limit=16)
        content = _clean_text(getattr(raw, "content", ""), limit=240)
        if role not in {"user", "assistant"} or not content:
            continue
        turns.append(ConversationTurnView(
            role=role,
            content=content,
            created_at=_safe_float(getattr(raw, "created_at", 0)),
        ))
    return tuple(turns)


def _recent_user_turns(memory: ConversationMemory) -> tuple[str, ...]:
    values = [
        _clean_text(getattr(raw, "content", ""), limit=180)
        for raw in list(memory.recent_turns or [])[-8:]
        if _clean_text(getattr(raw, "role", ""), limit=16) == "user"
    ]
    return tuple(value for value in values if value)[-4:]


def _previous_candidate_count(memory: ConversationMemory) -> int:
    history = [
        item for item in list(memory.search_history or [])
        if isinstance(item, dict)
    ]
    if not history and isinstance(memory.latest_search, dict):
        history = [memory.latest_search]
    if len(history) < 2:
        return 0
    current_at = (
        memory.latest_search.get("created_at")
        if isinstance(memory.latest_search, dict)
        else None
    )
    current_index = next(
        (
            index for index, item in enumerate(history)
            if item.get("created_at") == current_at
        ),
        len(history) - 1,
    )
    if current_index <= 0:
        return 0
    recipes = history[current_index - 1].get("recipes")
    return min(20, len(recipes)) if isinstance(recipes, list) else 0


def _view_load_state(
    snapshot: RuntimeMemorySnapshot | None,
) -> MemoryViewLoadState:
    if snapshot is None:
        return MemoryViewLoadState(provided=False)
    return MemoryViewLoadState(
        provided=True,
        short_term_status=snapshot.short_term_status,
        long_term_status=snapshot.long_term.status,
        profile_version=_safe_nonnegative_int(snapshot.long_term.profile_version),
        is_new_session=snapshot.is_new_session,
        hydrated_at=_safe_float(snapshot.hydrated_at),
    )


def build_safety_memory_view(
    snapshot: RuntimeMemorySnapshot | None,
    *,
    resolved_context: dict[str, Any] | None = None,
) -> SafetyMemoryView:
    """从本轮快照生成安全门禁视图；``None`` 保留 not-provided 语义。"""
    load_state = _view_load_state(snapshot)
    if snapshot is None:
        return SafetyMemoryView(
            load_state=load_state,
            policy_resolved=resolved_context is not None,
            effective_preferences=_preference_values(
                (resolved_context or {}).get("preferences")
            ),
            effective_temporal_dietary_constraints=_temporal_constraints(
                (resolved_context or {}).get(
                    "temporal_dietary_constraints"
                )
            ),
        )
    long_term = (
        _preference_values(snapshot.long_term.preferences)
        if load_state.long_term_available
        else PreferenceValuesView()
    )
    session = (
        _preference_values(snapshot.short_term.preferences)
        if load_state.short_term_available
        else PreferenceValuesView()
    )
    current_turn = _preference_values(snapshot.current_turn_preferences)
    invalidated = (
        _preference_values(snapshot.long_term.invalidated_preferences)
        if load_state.long_term_available
        else PreferenceValuesView()
    )
    return SafetyMemoryView(
        load_state=load_state,
        long_term_constraints=_safety_constraints(long_term),
        session_constraints=_safety_constraints(session),
        current_turn_constraints=_safety_constraints(current_turn),
        invalidated_long_term_constraints=_safety_constraints(invalidated),
        temporal_dietary_constraints=(
            _temporal_constraints(
                snapshot.long_term.temporal_dietary_constraints
            )
            if load_state.long_term_available
            else ()
        ),
        current_turn_mutations=_preference_mutations(
            snapshot.current_turn_preference_mutations
        ),
        policy_resolved=resolved_context is not None,
        effective_preferences=_preference_values(
            (resolved_context or {}).get("preferences")
        ),
        effective_temporal_dietary_constraints=_temporal_constraints(
            (resolved_context or {}).get("temporal_dietary_constraints")
        ),
    )


def build_planner_memory_view(
    snapshot: RuntimeMemorySnapshot | None,
    *,
    resolved_context: dict[str, Any] | None = None,
) -> PlannerMemoryView:
    """生成 Planner 最小视图；不会把 ConversationMemory 交给 Planner。"""
    load_state = _view_load_state(snapshot)
    if snapshot is None:
        return PlannerMemoryView(
            load_state=load_state,
            policy_resolved=resolved_context is not None,
            effective_preferences=_preference_values(
                (resolved_context or {}).get("preferences")
            ),
            effective_temporal_dietary_constraints=_temporal_constraints(
                (resolved_context or {}).get(
                    "temporal_dietary_constraints"
                )
            ),
        )
    return PlannerMemoryView(
        load_state=load_state,
        long_term_preferences=(
            _preference_values(snapshot.long_term.preferences)
            if load_state.long_term_available
            else PreferenceValuesView()
        ),
        session_preferences=(
            _preference_values(snapshot.short_term.preferences)
            if load_state.short_term_available
            else PreferenceValuesView()
        ),
        current_turn_preferences=_preference_values(
            snapshot.current_turn_preferences
        ),
        temporal_dietary_constraints=(
            _temporal_constraints(
                snapshot.long_term.temporal_dietary_constraints
            )
            if load_state.long_term_available
            else ()
        ),
        current_turn_mutations=_preference_mutations(
            snapshot.current_turn_preference_mutations
        ),
        recent_user_turns=(
            _text_values(
                resolved_context.get("recent_user_turns"),
                limit=4,
                text_limit=180,
            )
            if resolved_context is not None
            else (
                _recent_user_turns(snapshot.short_term)
                if load_state.short_term_available
                else ()
            )
        ),
        previous_candidate_count=(
            _safe_nonnegative_int(
                resolved_context.get("previous_candidate_count")
            )
            if resolved_context is not None
            else (
                _previous_candidate_count(snapshot.short_term)
                if load_state.short_term_available
                else 0
            )
        ),
        policy_resolved=resolved_context is not None,
        effective_preferences=_preference_values(
            (resolved_context or {}).get("preferences")
        ),
        effective_temporal_dietary_constraints=_temporal_constraints(
            (resolved_context or {}).get("temporal_dietary_constraints")
        ),
    )


def build_qa_memory_view(
    snapshot: RuntimeMemorySnapshot | None,
    *,
    resolved_context: dict[str, Any] | None = None,
) -> QAMemoryView:
    """生成 QA 数据视图；只保留允许进入问答上下文的字段。"""
    load_state = _view_load_state(snapshot)
    if snapshot is None:
        return QAMemoryView(
            load_state=load_state,
            policy_resolved=resolved_context is not None,
            effective_preferences=_preference_values(
                (resolved_context or {}).get("preferences")
            ),
            effective_temporal_dietary_constraints=_temporal_constraints(
                (resolved_context or {}).get(
                    "temporal_dietary_constraints"
                )
            ),
        )
    long_term_available = load_state.long_term_available
    short_term_available = load_state.short_term_available
    summary = (
        snapshot.short_term.summary
        if short_term_available and isinstance(snapshot.short_term.summary, dict)
        else {}
    )
    return QAMemoryView(
        load_state=load_state,
        long_term_preferences=(
            _preference_values(snapshot.long_term.preferences)
            if long_term_available
            else PreferenceValuesView()
        ),
        session_preferences=(
            _preference_values(snapshot.short_term.preferences)
            if short_term_available
            else PreferenceValuesView()
        ),
        current_turn_preferences=_preference_values(
            snapshot.current_turn_preferences
        ),
        temporal_dietary_constraints=(
            _temporal_constraints(
                snapshot.long_term.temporal_dietary_constraints
            )
            if long_term_available
            else ()
        ),
        current_turn_mutations=_preference_mutations(
            snapshot.current_turn_preference_mutations
        ),
        general_facts=(
            _general_profile_facts(snapshot.long_term.general_facts)
            if long_term_available
            else ()
        ),
        conversation_digest=_text_values(
            summary.get("conversation_digest"),
            limit=8,
            text_limit=240,
        ),
        recent_turns=(
            _recent_turns(snapshot.short_term)
            if short_term_available
            else ()
        ),
        policy_resolved=resolved_context is not None,
        effective_preferences=_preference_values(
            (resolved_context or {}).get("preferences")
        ),
        effective_temporal_dietary_constraints=_temporal_constraints(
            (resolved_context or {}).get("temporal_dietary_constraints")
        ),
        preferred_name=(
            _clean_text(
                (resolved_context or {}).get("preferred_name"),
                limit=160,
            )
            or None
            if resolved_context is not None
            else (
                _clean_text(snapshot.long_term.preferred_name, limit=160)
                or None
                if long_term_available
                else None
            )
        ),
        display_name=(
            _clean_text(
                (resolved_context or {}).get("display_name"),
                limit=160,
            )
            or None
            if resolved_context is not None
            else (
                _clean_text(snapshot.long_term.display_name, limit=160)
                or None
                if long_term_available
                else None
            )
        ),
        account_digest=(
            _text_values(
                (resolved_context or {}).get("account_digest"),
                limit=20,
                text_limit=240,
            )
            if resolved_context is not None
            else (
                _text_values(
                    snapshot.long_term.account_digest,
                    limit=20,
                    text_limit=240,
                )
                if long_term_available
                else ()
            )
        ),
    )


def _load_state_context(state: MemoryViewLoadState) -> dict[str, Any]:
    return {
        "provided": state.provided,
        "short_term_status": state.short_term_status,
        "long_term_status": state.long_term_status,
        "profile_version": state.profile_version,
        "is_new_session": state.is_new_session,
        "hydrated_at": state.hydrated_at,
    }


def _compat_load_state_context(
    state: MemoryViewLoadState,
) -> dict[str, Any]:
    """旧 Router/Planner 字段名适配；保留明确的来源状态。"""
    return {
        "memory_context_provided": state.provided,
        "short_term_memory_status": state.short_term_status,
        "long_term_memory_status": state.long_term_status,
        "profile_version": state.profile_version,
        "is_new_session": state.is_new_session,
        "hydrated_at": state.hydrated_at,
    }


def _preference_context(view: PreferenceValuesView) -> dict[str, list[str]]:
    return {
        "likes": list(view.likes),
        "dislikes": list(view.dislikes),
        "allergens": list(view.allergens),
        "dietary_constraints": list(view.dietary_constraints),
        "available_ingredients": list(view.available_ingredients),
    }


def _safety_context(view: SafetyConstraintsView) -> dict[str, list[str]]:
    return {
        "dislikes": list(view.dislikes),
        "allergens": list(view.allergens),
        "dietary_constraints": list(view.dietary_constraints),
    }


def _temporal_context(
    view: TemporalDietaryConstraintView,
) -> dict[str, Any]:
    return {
        "value": view.value,
        "source": view.source,
        "updated_at": view.updated_at,
        "expires_at": view.expires_at,
    }


def _mutation_context(view: PreferenceMutationView) -> dict[str, str]:
    return {
        "operation": view.operation,
        "bucket": view.bucket,
        "value": view.value,
        "scope": view.scope,
    }


def _profile_fact_context(view: GeneralProfileFactView) -> dict[str, Any]:
    return {
        "id": view.fact_id,
        "category": view.category,
        "key": view.key,
        "value": view.value,
        "subject": view.subject,
        "scope": view.scope,
        "source": view.source,
        "updated_at": view.updated_at,
        "expires_at": view.expires_at,
    }
