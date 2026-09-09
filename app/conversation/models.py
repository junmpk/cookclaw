"""与存储实现无关的会话记忆模型。"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ConversationTaskState:
    """与一段短期会话同生命周期的结构化业务状态。

    字段只保存已经由真实检索、用户明确选择或确定性设备流程产生的事实；
    不从助手自然语言回复反推业务状态。
    """

    schema_version: int = 1
    current_task: str | None = None
    candidate_recipes: list[dict[str, Any]] = field(default_factory=list)
    candidate_language: str | None = None
    candidate_decision_id: str = ""
    candidate_updated_at: float | None = None
    active_search_request: dict[str, Any] = field(default_factory=dict)
    # active search 的语义 TTL 只能由自身更新时间判定，不能借用
    # 整份 task state 的 updated_at，否则无关 patch 会让旧搜索复活。
    active_search_updated_at: float | None = None
    selected_recipe_id: str | None = None
    selected_recipe_name: str | None = None
    selected_recipe: dict[str, Any] | None = None
    selected_recipe_updated_at: float | None = None
    excluded_recipe_ids: list[str] = field(default_factory=list)
    excluded_recipe_updated_at: float | None = None
    focus: dict[str, Any] | None = None
    pending_search_clarification: dict[str, Any] | None = None
    # 用户资料修改与设备确认必须使用不同状态，避免一句“嗯”同时确认两件事。
    pending_profile_update: dict[str, Any] | None = None
    # 仅在 PostgreSQL 写入失败时保存本会话覆盖值；长期事实仍以画像表为准。
    temporary_profile: dict[str, Any] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    pending_device_start: dict[str, Any] | None = None
    # 用户确认后，pending 原子迁移到持久执行记录。它覆盖外部调用前后所有
    # crash window；未验证结果不得因新会话或进程重启而被当成“尚未执行”。
    device_execution: dict[str, Any] | None = None
    menu_task: dict[str, Any] | None = None
    selected_device_id: str | None = None
    active_cooking: dict[str, Any] | None = None
    language: str | None = None
    last_tool_result: dict[str, Any] | None = None
    updated_at: float = 0.0

    def has_data(self) -> bool:
        """是否包含需要持久化或迁移的业务事实。"""
        return any(
            (
                self.current_task,
                self.candidate_recipes,
                self.active_search_request,
                self.selected_recipe_id,
                self.excluded_recipe_ids,
                self.focus,
                self.pending_search_clarification,
                self.pending_profile_update,
                self.temporary_profile,
                self.pending_action,
                self.pending_device_start,
                self.device_execution,
                self.menu_task,
                self.active_cooking,
                self.last_tool_result,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConversationTaskState":
        raw = dict(data or {})
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in raw.items() if key in allowed})


@dataclass
class ConversationMemory:
    thread_id: str
    channel: str = "qq"

    user_id: str | None = None
    display_name: str | None = None
    preferred_name: str | None = None
    recent_turns: list[ConversationTurn] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, list[str]] = field(default_factory=lambda: {
        "likes": [],
        "dislikes": [],
        "allergens": [],
        "dietary_constraints": [],
        "available_ingredients": [],
    })
    latest_search: dict[str, Any] | None = None
    # 按发生顺序保留最近几轮搜索快照。latest_search 是当前正在讨论的那一轮，
    # 因此用户说“上一批”时可以移动游标，而不是把这句话当成搜索词。
    search_history: list[dict[str, Any]] = field(default_factory=list)
    # 账号/会话最近的真实饮食行为。只记录用户搜过或确认开火的菜，
    # 不根据助手文案猜偏好；由 ConversationService 做长度与过期控制。
    food_history: list[dict[str, Any]] = field(default_factory=list)
    # 仅账号画像使用：保存带来源和状态的长期事实。普通 thread 保持空数组，
    # 短期会话仍由 recent_turns / summary / preferences 等字段承载。
    long_term_facts: list[dict[str, Any]] = field(default_factory=list)
    # 候选、选择、排除、待确认和设备任务统一放在一个嵌套对象中，避免同一
    # 业务事实散落成多个 Redis 顶层字段。
    task_state: ConversationTaskState = field(default_factory=ConversationTaskState)
    # 1 = 升级前内嵌状态仍可作为迁移源；2 = 独立 task-state key 已接管，
    # 即使独立 key 之后过期，也绝不能从旧 JSON 镜像复活任务。
    task_state_storage_version: int = 1
    total_turns: int = 0
    turns_since_summary: int = 0
    estimated_tokens: int = 0
    version: int = 0
    # 会话重置/清除时递增；旧 worker 的写入即使 version 更大也不能跨代复活。
    conversation_generation: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMemory":
        raw = dict(data or {})
        raw["recent_turns"] = [
            item if isinstance(item, ConversationTurn) else ConversationTurn(**item)
            for item in raw.get("recent_turns", [])
            if isinstance(item, (dict, ConversationTurn))
        ]
        task_state = raw.get("task_state")
        if isinstance(task_state, ConversationTaskState):
            raw["task_state"] = task_state
        else:
            # 兼容升级前 Redis payload：旧版澄清状态位于顶层。
            legacy_clarification = raw.pop("pending_search_clarification", None)
            state = ConversationTaskState.from_dict(
                task_state if isinstance(task_state, dict) else None
            )
            if (
                state.pending_search_clarification is None
                and isinstance(legacy_clarification, dict)
            ):
                state.pending_search_clarification = dict(legacy_clarification)
            raw["task_state"] = state
        allowed = cls.__dataclass_fields__
        raw = {key: value for key, value in raw.items() if key in allowed}
        return cls(**raw)
