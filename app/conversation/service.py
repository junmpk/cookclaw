"""会话记忆服务：记录最近消息、搜索快照与阈值摘要。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import time
import weakref
from copy import deepcopy
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from app.conversation.context_builder import build_conversation_context
from app.conversation.models import (
    ConversationMemory,
    ConversationTaskState,
    ConversationTurn,
)
from app.conversation.preference_parser import (
    PreferenceMutation,
    apply_preference_mutations,
    extract_preference_mutations,
)
from app.conversation.profile_facts import (
    ProfileFactApplyResult,
    ProfileFactCandidate,
    active_general_profile_facts,
    apply_profile_fact_mutations,
)
from app.conversation.runtime_memory import (
    LongTermMemorySnapshot,
    MemoryScope,
    PlannerMemoryView,
    QAMemoryView,
    RuntimeMemorySnapshot,
    SafetyMemoryView,
    ShortTermLoadStatus,
    ShortTermRuntimeSnapshot,
    build_planner_memory_view,
    build_qa_memory_view,
    build_safety_memory_view,
)
from app.conversation.store import ConversationStoreConflict, InMemoryConversationStore
from app.conversation.summarizer import sanitize_preferences, summarize_incrementally, update_preferences
from app.conversation.task_state_store import (
    DeviceStartClaim,
    TaskStateRecord,
    TaskStateStoreConflict,
    refresh_task_state_derived,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_ACTIVE_SEARCH_TTL_SECONDS = 21_600


class ConversationBackendUnavailable(RuntimeError):
    """会话后端已超时或处于短时熔断，调用方应立即降级。"""


class PreferredNameConflict(RuntimeError):
    """用户称呼在确认期间已被其他请求修改，拒绝覆盖新值。"""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """一轮会话的受控身份与存储能力。

    ``session_memory`` 只代表可按稳定 thread_id 使用 Redis 短期会话；
    ``account_profile`` 才代表可映射到 PostgreSQL 账号画像。两者必须分开，
    避免匿名 Web session 因获得上下文能力而被误当成 QQ 账号。
    """

    thread_id: str
    channel: str
    user_id: str | None
    session_memory: bool
    account_profile: bool


_PERSISTENT_PREFERENCE_KEYS = (
    "likes",
    "dislikes",
    "allergens",
    "dietary_constraints",
)
_PREFERENCE_FACT_BUCKETS = {
    "preference_like": "likes",
    "preference_dislike": "dislikes",
    "allergy": "allergens",
    "dietary_constraint": "dietary_constraints",
}
_TRANSIENT_DIETARY_CONSTRAINTS = {
    "减肥",
    "减脂",
    "低脂",
    "控卡",
    "weight loss",
    "low-fat",
    "low fat",
}
_EXACT_HOW_TO_RE = re.compile(
    r"怎么做|如何做|如何制作|做法|详细步骤|具体步骤|"
    r"how\s+to\s+(?:make|cook)|recipe\s+steps",
    flags=re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"这个|这道|那个|那道|它|刚才|上次|前面|第[一二三四五六七八九十\d]+道|"
    r"\b(?:this|that|it|previous|earlier|last time)\b",
    flags=re.IGNORECASE,
)


def persistent_preferences_from_text(text: str) -> dict[str, list[str]]:
    """只提取用户本轮明确声明的稳定饮食事实。"""
    mutations = [
        mutation
        for mutation in extract_preference_mutations(str(text or "").strip())
        if (
            mutation.scope == "long_term_candidate"
            or (
                mutation.scope == "temporal"
                and mutation.bucket == "dietary_constraints"
            )
        )
    ]
    extracted, _changed = apply_preference_mutations({}, mutations)
    extracted = sanitize_preferences(extracted)
    return {
        key: list(extracted.get(key) or [])
        for key in _PERSISTENT_PREFERENCE_KEYS
    }


def is_transient_dietary_constraint(value: str) -> bool:
    """阶段性饮食目标不能按永久硬约束使用。"""
    return str(value or "").strip().lower() in _TRANSIENT_DIETARY_CONSTRAINTS


def should_persist_exchange(user_text: str) -> bool:
    """账号画像升级策略；不应再用它决定短期 transcript 是否连续。"""
    text = " ".join(str(user_text or "").split()).strip()
    if not text:
        return False
    if any(persistent_preferences_from_text(text).values()):
        return True
    if len(text) <= 100 and _EXACT_HOW_TO_RE.search(text) and not _REFERENCE_RE.search(text):
        return False
    return True


def should_persist_session_exchange(user_text: str) -> bool:
    """稳定 session 保存每个非空用户轮次，保证后续指代和语气能承接。"""
    return bool(" ".join(str(user_text or "").split()).strip())


class ConversationService:
    def __init__(self, store, *, profile_store=None,
                 profile_memory_index=None,
                 profile_fact_extractor=None,
                 ttl_seconds: int = 86400, max_turns: int = 20,
                 keep_recent_turns: int = 6, max_tokens: int = 6000,
                 profile_ttl_seconds: int = 31_536_000,
                 transient_diet_ttl_seconds: int = 2_592_000,
                 memory_recall_top_k: int = 8,
                 memory_recall_min_score: float = 0.35,
                 read_timeout_seconds: float = 5.0,
                 write_timeout_seconds: float = 5.0,
                 circuit_failure_threshold: int = 2,
                 circuit_cooldown_seconds: float = 30.0) -> None:
        self.store = store
        # 单元测试可让会话和画像共用内存 Store；生产使用 Redis + PostgreSQL。
        self.profile_store = profile_store or store
        # PostgreSQL 是事实源；该索引仅用于用户内相关性召回，可安全降级。
        self.profile_memory_index = profile_memory_index
        self.profile_fact_extractor = profile_fact_extractor
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.keep_recent_messages = max(2, keep_recent_turns * 2)
        self.max_tokens = max_tokens
        self.profile_ttl_seconds = profile_ttl_seconds
        self.transient_diet_ttl_seconds = max(
            1,
            int(transient_diet_ttl_seconds),
        )
        self.memory_recall_top_k = max(1, int(memory_recall_top_k))
        self.memory_recall_min_score = max(
            -1.0,
            min(1.0, float(memory_recall_min_score)),
        )
        self.read_timeout_seconds = max(0.05, float(read_timeout_seconds))
        self.write_timeout_seconds = max(0.05, float(write_timeout_seconds))
        self.circuit_failure_threshold = max(1, int(circuit_failure_threshold))
        self.circuit_cooldown_seconds = max(1.0, float(circuit_cooldown_seconds))
        # Web 会话 ID 是高熵且可能一次性使用。弱引用锁表能在没有持有者或
        # 等待者后自动回收，避免长驻进程为每个匿名 thread 永久保留字典键。
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._turn_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._backend_failures: dict[str, int] = {}
        self._backend_open_until: dict[str, float] = {}
        self._pending_tasks: set[asyncio.Task] = set()
        self._task_tails: dict[str, asyncio.Task] = {}
        self._profile_task_tails: dict[str, asyncio.Task] = {}
        # 通道入口按 user -> assistant 保序入队。用户明确声明先落 Redis，
        # 长期画像等助手轮次完成后再异步更新，避免 PostgreSQL 阻塞本轮回复。
        self._pending_profile_texts: dict[str, list[str]] = {}
        self._handled_general_memory_digests: dict[str, set[str]] = {}

    def _lock(self, thread_id: str) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def turn_lock(self, thread_id: str) -> asyncio.Lock:
        """同一进程内按会话串行完整 turn；与单次存储写锁分离。"""
        return self._turn_locks.setdefault(thread_id, asyncio.Lock())

    @staticmethod
    def _is_availability_error(exc: Exception) -> bool:
        return isinstance(exc, (TimeoutError, OSError, ConversationBackendUnavailable)) or (
            type(exc).__name__ in {"TimeoutError", "ConnectionError"}
        )

    async def _call_backend(
        self,
        backend_name: str,
        operation: str,
        factory: Callable[[], Awaitable[_T]],
        *,
        timeout_seconds: float,
    ) -> _T:
        now = time.monotonic()
        open_until = self._backend_open_until.get(backend_name, 0.0)
        if open_until > now:
            raise ConversationBackendUnavailable(
                f"{backend_name} circuit open for {open_until - now:.2f}s"
            )

        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await factory()
        except Exception as exc:
            elapsed_ms = (time.monotonic() - started) * 1000
            if self._is_availability_error(exc):
                failures = self._backend_failures.get(backend_name, 0) + 1
                self._backend_failures[backend_name] = failures
                if failures >= self.circuit_failure_threshold:
                    self._backend_open_until[backend_name] = (
                        time.monotonic() + self.circuit_cooldown_seconds
                    )
            logger.warning(
                "会话后端操作失败: backend=%s operation=%s elapsed_ms=%.1f error_type=%s",
                backend_name,
                operation,
                elapsed_ms,
                type(exc).__name__,
            )
            raise
        self._backend_failures.pop(backend_name, None)
        self._backend_open_until.pop(backend_name, None)
        return result

    async def _load_conversation(self, thread_id: str) -> ConversationMemory | None:
        return await self._call_backend(
            "conversation",
            "load",
            lambda: self.store.load(thread_id),
            timeout_seconds=self.read_timeout_seconds,
        )

    async def _save_conversation(self, memory: ConversationMemory, expires_at: float) -> None:
        await self._call_backend(
            "conversation",
            "save",
            lambda: self.store.save(memory, expires_at=expires_at),
            timeout_seconds=self.write_timeout_seconds,
        )

    async def _delete_conversation(self, thread_id: str) -> None:
        await self._call_backend(
            "conversation",
            "delete",
            lambda: self.store.delete(thread_id),
            timeout_seconds=self.write_timeout_seconds,
        )

    async def _reset_conversation(
        self,
        memory: ConversationMemory,
        *,
        expected_version: int,
        expected_generation: int,
    ) -> None:
        reset = getattr(self.store, "reset", None)
        if reset is None:
            raise ConversationBackendUnavailable(
                "conversation store does not support reset tombstones"
            )
        now = time.time()
        await self._call_backend(
            "conversation",
            "reset",
            lambda: reset(
                memory,
                expected_version=max(0, int(expected_version)),
                expected_generation=max(0, int(expected_generation)),
                expires_at=now + self.ttl_seconds,
            ),
            timeout_seconds=self.write_timeout_seconds,
        )

    async def _load_profile(self, profile_key: str) -> ConversationMemory | None:
        return await self._call_backend(
            "profile",
            "load",
            lambda: self.profile_store.load(profile_key),
            timeout_seconds=self.read_timeout_seconds,
        )

    async def _save_profile(self, memory: ConversationMemory, expires_at: float) -> None:
        await self._call_backend(
            "profile",
            "save",
            lambda: self.profile_store.save(memory, expires_at=expires_at),
            timeout_seconds=self.write_timeout_seconds,
        )

    async def load(self, thread_id: str, *, channel: str | None = None, user_id: str | None = None) -> ConversationMemory:
        identity = self._resolve_runtime_identity(
            thread_id,
            channel=channel,
            user_id=user_id,
        )
        value = identity.thread_id
        resolved_channel = identity.channel
        resolved_user_id = identity.user_id
        memory = await self._load_conversation(value)
        if memory is None:
            memory = ConversationMemory(
                thread_id=value,
                channel=resolved_channel,
                user_id=resolved_user_id,
                task_state_storage_version=2,
            )
        elif identity.session_memory:
            identity_mismatch = bool(
                memory.thread_id != value
                or (
                    str(memory.channel or "").strip()
                    and memory.channel != resolved_channel
                )
                or (
                    memory.user_id
                    and memory.user_id != resolved_user_id
                )
            )
            if identity_mismatch:
                raise ValueError(
                    "conversation payload identity does not match storage key"
                )
            memory.thread_id = value
            memory.channel = resolved_channel
            memory.user_id = resolved_user_id
        elif resolved_user_id and not memory.user_id:
            memory.user_id = resolved_user_id
        memory.preferences = sanitize_preferences(memory.preferences)
        return memory

    @staticmethod
    def _safe_timestamp(value: object, *, default: float = 0.0) -> float:
        """持久化资料异常时局部丢弃坏时间值，不能拖垮整轮记忆加载。"""
        try:
            parsed = float(value or 0)
        except (TypeError, ValueError):
            return float(default)
        return parsed if math.isfinite(parsed) else float(default)

    @staticmethod
    def _safe_nonnegative_int(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _active_runtime_profile_facts(
        facts: list[dict],
        *,
        now: float | None = None,
    ) -> list[dict]:
        """只把当前有效事实装入单轮快照，排除清除、撤销和过期记录。"""
        current = time.time() if now is None else float(now)
        active: list[dict] = []
        for item in facts or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "active") != "active":
                continue
            raw_expires_at = item.get("expires_at")
            try:
                expires_at = float(raw_expires_at or 0)
            except (TypeError, ValueError):
                # 阶段事实的 TTL 无法解释时 fail closed，不能误当永久事实。
                continue
            if not math.isfinite(expires_at):
                continue
            if expires_at and expires_at <= current:
                continue
            if not str(item.get("value") or "").strip():
                continue
            normalized = deepcopy(item)
            for timestamp_key in (
                "created_at",
                "updated_at",
                "last_confirmed_at",
                "expires_at",
            ):
                if timestamp_key in normalized:
                    normalized[timestamp_key] = (
                        ConversationService._safe_timestamp(
                            normalized.get(timestamp_key)
                        )
                    )
            active.append(normalized)
        return active[-100:]

    def _long_term_runtime_snapshot(
        self,
        profile: ConversationMemory | None,
        *,
        loaded_at: float,
        status: str | None = None,
    ) -> LongTermMemorySnapshot:
        """从 PostgreSQL 行生成纯读投影；不回填 thread，也不写任何存储。"""
        if profile is None:
            return LongTermMemorySnapshot(
                status="empty" if status is None else status,
                loaded_at=loaded_at,
            )

        raw_facts = [
            item for item in list(profile.long_term_facts or [])
            if isinstance(item, dict)
        ]
        active_facts = self._active_runtime_profile_facts(
            raw_facts,
            now=loaded_at,
        )
        profile_preferences = sanitize_preferences(profile.preferences)
        # 可用食材描述的是某一顿/某个 thread 的现场条件。即使旧 PG 行因历史
        # 版本残留该字段，也不能把“上次冰箱里有什么”带进新会话。
        profile_preferences["available_ingredients"] = []
        active_preference_signatures = {
            (
                str(item.get("type") or ""),
                str(item.get("value") or "").strip().lower(),
            )
            for item in active_facts
            if (
                str(item.get("type") or "") in _PREFERENCE_FACT_BUCKETS
                and str(item.get("value") or "").strip()
            )
        }
        invalidated_preferences = {
            key: [] for key in _PERSISTENT_PREFERENCE_KEYS
        }
        for item in raw_facts:
            fact_type = str(item.get("type") or "")
            bucket = _PREFERENCE_FACT_BUCKETS.get(fact_type)
            value = str(item.get("value") or "").strip()
            if not bucket or not value:
                continue
            raw_expires_at = item.get("expires_at")
            try:
                expires_at = float(raw_expires_at or 0)
                invalid_expiry = not math.isfinite(expires_at)
            except (TypeError, ValueError):
                expires_at = 0.0
                invalid_expiry = True
            invalidated = (
                str(item.get("status") or "active") != "active"
                or invalid_expiry
                or (expires_at > 0 and expires_at <= loaded_at)
            )
            if (
                invalidated
                and (fact_type, value.lower())
                not in active_preference_signatures
                and value not in invalidated_preferences[bucket]
            ):
                invalidated_preferences[bucket].append(value)
        for bucket, values in invalidated_preferences.items():
            if not values:
                continue
            blocked = {value.lower() for value in values}
            profile_preferences[bucket] = [
                value
                for value in profile_preferences[bucket]
                if value.lower() not in blocked
            ]
        stable_preferences, temporal = self._separate_temporal_dietary_constraints(
            profile_preferences,
            active_facts,
            fallback_updated_at=self._safe_timestamp(
                profile.updated_at,
                default=loaded_at,
            ),
        )
        digest = self._canonical_account_digest(stable_preferences, temporal)
        preferred_name = str(profile.preferred_name or "").strip() or None
        display_name = str(profile.display_name or "").strip() or None
        events = [
            deepcopy(item)
            for item in list(profile.food_history or [])[-20:]
            if isinstance(item, dict)
        ]
        general_facts = active_general_profile_facts(active_facts, limit=50)
        has_data = bool(
            preferred_name
            or display_name
            or any(stable_preferences.values())
            or temporal
            or digest
            or events
            or general_facts
        )
        return LongTermMemorySnapshot(
            status=(status or ("loaded" if has_data else "empty")),
            profile_version=self._safe_nonnegative_int(profile.version),
            preferred_name=preferred_name,
            display_name=display_name,
            preferences=deepcopy(stable_preferences),
            temporal_dietary_constraints=deepcopy(temporal),
            account_digest=list(digest),
            events=events,
            invalidated_preferences=invalidated_preferences,
            facts=active_facts,
            general_facts=deepcopy(general_facts),
            food_memory_cleared_at=self._safe_timestamp(
                (
                    profile.summary
                    if isinstance(profile.summary, dict)
                    else {}
                ).get("food_memory_cleared_at")
            ),
            loaded_at=loaded_at,
        )

    @staticmethod
    def _resolve_runtime_identity(
        thread_id: str,
        *,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> RuntimeIdentity:
        value = str(thread_id or "")
        prefix = value.split(":", 1)[0].lower()
        session_memory = supports_session_memory(value)
        account_profile = supports_account_profile(value)

        if prefix == "web":
            if channel is not None and channel != "web":
                raise ValueError("runtime memory channel does not match thread id")
            if user_id is not None:
                raise ValueError("web session memory cannot accept an account user id")
            return RuntimeIdentity(
                thread_id=value,
                channel="web",
                user_id=None,
                session_memory=session_memory,
                account_profile=False,
            )

        inferred_channel, inferred_user_id = conversation_identity(value)
        if session_memory:
            if channel is not None and channel != inferred_channel:
                raise ValueError("runtime memory channel does not match thread id")
            if user_id is not None and user_id != inferred_user_id:
                raise ValueError("runtime memory user does not match thread id")
            # 账号隔离只信任受控 thread_id，不信任 Redis payload 中可变字段。
            resolved_channel = inferred_channel
            resolved_user_id = inferred_user_id
        else:
            resolved_channel = channel or (
                prefix if prefix in {"qq", "whatsapp", "weixin", "web"}
                else inferred_channel
            )
            resolved_user_id = user_id or inferred_user_id
        return RuntimeIdentity(
            thread_id=value,
            channel=resolved_channel,
            user_id=resolved_user_id,
            session_memory=session_memory,
            account_profile=account_profile,
        )

    @staticmethod
    def _task_state_has_data(state: ConversationTaskState) -> bool:
        return state.has_data()

    @staticmethod
    def _project_task_state_for_runtime(
        state: ConversationTaskState,
        *,
        now: float | None = None,
    ) -> ConversationTaskState:
        """构造不修改存储的语义 TTL 视图，并为旧 payload 补齐字段时间。"""
        projected = ConversationTaskState.from_dict(state.to_dict())
        observed_at = float(
            projected.active_search_updated_at
            or projected.updated_at
            or 0
        )
        if projected.active_search_request:
            projected.active_search_updated_at = observed_at or None
            if (
                observed_at
                and float(now if now is not None else time.time()) - observed_at
                > _ACTIVE_SEARCH_TTL_SECONDS
            ):
                projected.active_search_request = {}
                projected.active_search_updated_at = None
                # 过期投影不推进 updated_at，但 current_task 不得继续指向旧搜索。
                refresh_task_state_derived(
                    projected,
                    now=float(projected.updated_at or 0),
                )
        return projected

    async def load_short_term_runtime(
        self,
        thread_id: str,
        *,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> ShortTermRuntimeSnapshot:
        """只读一次 Redis 短期会话，并保留 new/unavailable 真实语义。"""
        identity = self._resolve_runtime_identity(
            thread_id,
            channel=channel,
            user_id=user_id,
        )
        value = identity.thread_id
        resolved_channel = identity.channel
        resolved_user_id = identity.user_id

        task_state_persistent = supports_persistent_task_state(value)
        if not task_state_persistent:
            short = ConversationMemory(
                thread_id=value,
                channel=resolved_channel,
                user_id=resolved_user_id,
                task_state_storage_version=2,
            )
            return ShortTermRuntimeSnapshot(
                thread_id=value,
                channel=resolved_channel,
                user_id=resolved_user_id,
                memory=short,
                status="unsupported",
                is_new_session=None,
                task_state_status="unsupported",
                task_state_revision=0,
                task_state_generation=0,
            )

        short_status: ShortTermLoadStatus = (
            "loaded" if identity.session_memory else "unsupported"
        )
        is_new_session: bool | None = (
            False if identity.session_memory else None
        )
        if identity.session_memory:
            try:
                persisted = await self._load_conversation(value)
            except Exception as exc:
                short_status = "unavailable"
                is_new_session = None
                logger.warning(
                    "运行时短期记忆读取失败，本轮使用空短期态: error_type=%s",
                    type(exc).__name__,
                )
                persisted = None
        else:
            persisted = None
        if persisted is None:
            if identity.session_memory and short_status != "unavailable":
                short_status = "new"
                is_new_session = True
            short = ConversationMemory(
                thread_id=value,
                channel=resolved_channel,
                user_id=resolved_user_id,
            )
        else:
            identity_mismatch = bool(
                persisted.thread_id != value
                or (
                    str(persisted.channel or "").strip()
                    and persisted.channel != resolved_channel
                )
                or (
                    persisted.user_id
                    and persisted.user_id != resolved_user_id
                )
            )
            if identity_mismatch:
                logger.warning(
                    "短期记忆身份与 key 不一致，本轮丢弃该 payload"
                )
                short_status = "invalid"
                is_new_session = None
                short = ConversationMemory(
                    thread_id=value,
                    channel=resolved_channel,
                    user_id=resolved_user_id,
                    task_state_storage_version=2,
                )
            else:
                short = ConversationMemory.from_dict(persisted.to_dict())
                short.thread_id = value
                short.channel = resolved_channel
                short.user_id = resolved_user_id
        short.preferences = sanitize_preferences(short.preferences)
        task_state_status = "new"
        task_state_revision = 0
        task_state_generation = 0
        load_task_record = getattr(self.store, "load_task_state_record", None)
        if load_task_record is None:
            task_state_status = "legacy"
        else:
            try:
                task_record = await self._call_backend(
                    "conversation",
                    "load_task_state_record",
                    lambda: load_task_record(value),
                    timeout_seconds=self.read_timeout_seconds,
                )
            except Exception as exc:
                logger.warning(
                    "独立任务状态读取失败，本轮不得回退进程缓存: error_type=%s",
                    type(exc).__name__,
                )
                task_state_status = "unavailable"
            else:
                if task_record is not None:
                    if task_record.thread_id != value:
                        task_state_status = "invalid"
                    else:
                        short.task_state = ConversationTaskState.from_dict(
                            task_record.state.to_dict()
                        )
                        task_state_revision = max(0, int(task_record.revision))
                        task_state_generation = max(
                            0,
                            int(task_record.generation),
                        )
                        task_state_status = "loaded"
                elif short_status in {"unavailable", "invalid"}:
                    # 独立记录尚不存在时，必须先确认旧 ConversationMemory 是否
                    # 含迁移源；分裂故障下不能把“读不到旧状态”误判成空状态。
                    task_state_status = short_status
                elif (
                    short.task_state_storage_version < 2
                    and self._task_state_has_data(short.task_state)
                ):
                    # 升级前嵌在 ConversationMemory 中的状态只作为一次性迁移源；
                    # 新链路首次修改时会以 revision=0 写入独立 key。
                    task_state_status = "legacy"
                elif short.task_state_storage_version >= 2:
                    # 独立 key 过期代表任务态自然失效；迁移后的旧镜像不是回退源。
                    short.task_state = ConversationTaskState()
        if task_state_status in {"unavailable", "invalid"}:
            # 状态标签必须保留故障语义，但 payload 必须失败关闭。否则
            # marker=2 的 conversation 兼容镜像仍可被读成旧候选或临时称呼。
            short.task_state = ConversationTaskState()
        short.task_state = self._project_task_state_for_runtime(short.task_state)
        return ShortTermRuntimeSnapshot(
            thread_id=value,
            channel=resolved_channel,
            user_id=resolved_user_id,
            memory=short,
            status=short_status,
            is_new_session=is_new_session,
            task_state_status=task_state_status,
            task_state_revision=task_state_revision,
            task_state_generation=task_state_generation,
        )

    async def load_runtime_memory(
        self,
        thread_id: str,
        *,
        channel: str | None = None,
        user_id: str | None = None,
        current_message: str = "",
        short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
    ) -> RuntimeMemorySnapshot:
        """一次组合短期会话和长期画像，供本轮所有处理器共享。

        读取失败与“没有资料”必须可区分。该方法只读，不会因为新 thread 而创建
        Redis key，也不会把 PostgreSQL 画像复制到短期会话。task scope 已经读取
        短期态时可传入隔离快照，避免同一轮重复读取 Redis。
        """
        hydrated_at = time.time()
        identity = self._resolve_runtime_identity(
            thread_id,
            channel=channel,
            user_id=user_id,
        )
        value = identity.thread_id
        resolved_channel = identity.channel
        resolved_user_id = identity.user_id
        current_text = str(current_message or "")
        current_turn_mutations = tuple(
            extract_preference_mutations(current_text)
        )
        current_turn_preferences = update_preferences({}, current_text)

        if short_term_snapshot is None:
            short_term_snapshot = await self.load_short_term_runtime(
                value,
                channel=resolved_channel,
                user_id=resolved_user_id,
            )
        elif not short_term_snapshot.belongs_to(value):
            raise ValueError("short-term runtime snapshot belongs to another thread")

        short = ConversationMemory.from_dict(
            short_term_snapshot.memory.to_dict()
        )
        short.thread_id = value
        short.channel = resolved_channel
        short.user_id = resolved_user_id
        short.preferences = sanitize_preferences(short.preferences)
        short_status = short_term_snapshot.status
        is_new_session = short_term_snapshot.is_new_session

        if not identity.session_memory:
            return RuntimeMemorySnapshot(
                thread_id=value,
                channel=resolved_channel,
                user_id=resolved_user_id,
                short_term=short,
                short_term_status="unsupported",
                long_term=LongTermMemorySnapshot(
                    status="unsupported",
                    loaded_at=hydrated_at,
                ),
                current_turn_preferences=current_turn_preferences,
                current_turn_preference_mutations=current_turn_mutations,
                is_new_session=None,
                hydrated_at=hydrated_at,
            )

        if not identity.account_profile:
            long_term = LongTermMemorySnapshot(
                status="unsupported",
                loaded_at=hydrated_at,
            )
        else:
            profile_key = self._profile_key(short)
            if not profile_key:
                raise ValueError("account profile identity is unavailable")
            try:
                profile = await self._load_profile(profile_key)
            except Exception as exc:
                logger.warning(
                    "运行时长期画像读取失败，本轮不使用长期资料: error_type=%s",
                    type(exc).__name__,
                )
                long_term = LongTermMemorySnapshot(
                    status="unavailable",
                    loaded_at=hydrated_at,
                )
            else:
                try:
                    long_term = self._long_term_runtime_snapshot(
                        profile,
                        loaded_at=hydrated_at,
                    )
                except Exception as exc:
                    logger.warning(
                        "运行时长期画像投影失败，本轮不使用长期资料: error_type=%s",
                        type(exc).__name__,
                    )
                    long_term = LongTermMemorySnapshot(
                        status="invalid",
                        loaded_at=hydrated_at,
                    )

        return RuntimeMemorySnapshot(
            thread_id=value,
            channel=short.channel,
            user_id=short.user_id,
            short_term=short,
            short_term_status=short_status,
            long_term=long_term,
            current_turn_preferences=current_turn_preferences,
            current_turn_preference_mutations=current_turn_mutations,
            is_new_session=is_new_session,
            hydrated_at=hydrated_at,
        )

    @staticmethod
    def _runtime_profile_preferences(
        snapshot: RuntimeMemorySnapshot,
    ) -> dict[str, list[str]]:
        preferences = sanitize_preferences(snapshot.long_term.preferences)
        temporal_values = [
            str(item.get("value") or "").strip()
            for item in snapshot.long_term.temporal_dietary_constraints
            if isinstance(item, dict) and str(item.get("value") or "").strip()
        ]
        for value in temporal_values:
            if value not in preferences["dietary_constraints"]:
                preferences["dietary_constraints"].append(value)
        return preferences

    def _account_context_from_runtime(
        self,
        snapshot: RuntimeMemorySnapshot,
    ) -> dict:
        """按既有优先级合并快照两层数据，不修改快照或持久化存储。"""
        memory = snapshot.short_term
        long_term = snapshot.long_term
        has_profile_identity = self._profile_key(memory) is not None
        if has_profile_identity and long_term.available:
            profile_preferences = self._runtime_profile_preferences(snapshot)
            events = [deepcopy(item) for item in long_term.events]
            facts = [deepcopy(item) for item in long_term.facts]
            cleared_at = float(long_term.food_memory_cleared_at or 0)
            preferred_name = long_term.preferred_name
            display_name = long_term.display_name
        elif has_profile_identity:
            profile_preferences = sanitize_preferences({})
            events = []
            facts = []
            cleared_at = 0.0
            preferred_name = None
            display_name = None
        else:
            profile_preferences = sanitize_preferences({})
            events = [deepcopy(item) for item in memory.food_history or []]
            facts = [deepcopy(item) for item in memory.long_term_facts or []]
            cleared_at = 0.0
            preferred_name = None
            display_name = None

        if has_profile_identity:
            legacy_events = [
                self._search_snapshot_event(item)
                for item in (memory.search_history or [])
                if (
                    isinstance(item, dict)
                    and self._safe_timestamp(item.get("created_at"))
                    > cleared_at
                )
            ]
            combined = sorted(
                [*events, *legacy_events],
                key=lambda item: self._safe_timestamp(item.get("created_at")),
            )
            events = []
            for event in combined:
                events = self._append_food_event(events, event)

        session_preferences = sanitize_preferences(memory.preferences)
        if has_profile_identity and long_term.available:
            for bucket, invalidated in (
                long_term.invalidated_preferences or {}
            ).items():
                if bucket not in session_preferences:
                    continue
                blocked = {
                    str(value).strip().lower() for value in invalidated
                }
                session_preferences[bucket] = [
                    value
                    for value in session_preferences[bucket]
                    if str(value).strip().lower() not in blocked
                ]
            if (
                long_term.food_memory_cleared_at > 0
                and self._safe_timestamp(memory.created_at)
                <= long_term.food_memory_cleared_at
            ):
                session_preferences = sanitize_preferences(
                    snapshot.current_turn_preferences
                )
        merged_preferences = self._merge_preferences(
            profile_preferences,
            session_preferences,
        )
        current_mutations = [
            item
            for item in snapshot.current_turn_preference_mutations
            if isinstance(item, PreferenceMutation)
        ]
        if current_mutations:
            merged_preferences, _changed = apply_preference_mutations(
                merged_preferences,
                current_mutations,
            )
            merged_preferences = sanitize_preferences(merged_preferences)
        merged_preferences, temporal = self._separate_temporal_dietary_constraints(
            merged_preferences,
            facts,
            fallback_updated_at=self._safe_timestamp(
                memory.updated_at,
                default=snapshot.hydrated_at,
            ),
        )
        return {
            "preferred_name": preferred_name,
            "display_name": display_name,
            "preferences": merged_preferences,
            "temporal_dietary_constraints": temporal,
            "account_digest": self._canonical_account_digest(
                merged_preferences,
                temporal,
            ),
            "events": events[-20:],
            "long_term_memory_status": long_term.status,
            "profile_version": long_term.profile_version,
        }

    def _preference_context_from_runtime(
        self,
        snapshot: RuntimeMemorySnapshot,
    ) -> dict:
        """精确任务只看账号长期饮食约束，不把通用画像送进 Planner。"""
        long_term = snapshot.long_term
        if long_term.available:
            preferences = self._runtime_profile_preferences(snapshot)
            preferred_name = long_term.preferred_name
            display_name = long_term.display_name
        else:
            preferences = sanitize_preferences({})
            preferred_name = None
            display_name = None
        current_mutations = [
            item
            for item in snapshot.current_turn_preference_mutations
            if isinstance(item, PreferenceMutation)
        ]
        if current_mutations:
            preferences, _changed = apply_preference_mutations(
                preferences,
                current_mutations,
            )
        preferences, temporal = self._separate_temporal_dietary_constraints(
            sanitize_preferences(preferences),
            long_term.facts if long_term.available else [],
            fallback_updated_at=snapshot.hydrated_at,
        )
        return {
            "preferred_name": preferred_name,
            "display_name": display_name,
            "preferences": preferences,
            "temporal_dietary_constraints": temporal,
            "account_digest": [],
            "events": [],
            "long_term_memory_status": long_term.status,
            "profile_version": long_term.profile_version,
        }

    @staticmethod
    def _assert_runtime_belongs_to(
        thread_id: str,
        snapshot: RuntimeMemorySnapshot,
    ) -> None:
        if not snapshot.belongs_to(thread_id):
            raise ValueError("runtime memory snapshot belongs to another thread")

    def _safety_memory_view_from_runtime(
        self,
        snapshot: RuntimeMemorySnapshot,
    ) -> SafetyMemoryView:
        """应用稳定偏好策略后生成 Router 安全视图。"""
        return build_safety_memory_view(
            snapshot,
            resolved_context=self._preference_context_from_runtime(snapshot),
        )

    def _planner_memory_view_from_runtime(
        self,
        snapshot: RuntimeMemorySnapshot,
    ) -> PlannerMemoryView:
        """应用长期/会话/本轮覆盖策略后生成 Planner 视图。"""
        resolved_context = build_conversation_context(
            snapshot.short_term,
            self._account_context_from_runtime(snapshot),
            include_recent=False,
        ).to_dict()
        previous_search = self._previous_search_snapshot(snapshot.short_term)
        resolved_context["previous_candidate_count"] = len(
            (previous_search or {}).get("recipes") or []
        )
        return build_planner_memory_view(
            snapshot,
            resolved_context=resolved_context,
        )

    def _qa_memory_view_from_runtime(
        self,
        snapshot: RuntimeMemorySnapshot,
        *,
        current_message: str = "",
    ) -> QAMemoryView:
        """生成 QA 视图；临时称呼覆盖仍由统一上下文规则解析。"""
        account_context = self._account_context_from_runtime(snapshot)
        resolved_context = build_conversation_context(
            snapshot.short_term,
            account_context,
            current_message=current_message,
            include_recent=True,
        ).to_dict()
        return build_qa_memory_view(
            snapshot,
            resolved_context=resolved_context,
        )

    async def safety_memory_view(
        self,
        thread_id: str,
        *,
        current_message: str = "",
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> SafetyMemoryView:
        """返回只含 Router 安全与稳定偏好决策所需数据的冻结视图。"""
        snapshot = runtime_memory or await self.load_runtime_memory(
            thread_id,
            current_message=current_message,
        )
        self._assert_runtime_belongs_to(thread_id, snapshot)
        return self._safety_memory_view_from_runtime(snapshot)

    async def planner_memory_view(
        self,
        thread_id: str,
        *,
        current_message: str = "",
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> PlannerMemoryView:
        """返回 Planner 的冻结视图，不暴露 ConversationMemory。"""
        snapshot = runtime_memory or await self.load_runtime_memory(
            thread_id,
            current_message=current_message,
        )
        self._assert_runtime_belongs_to(thread_id, snapshot)
        return self._planner_memory_view_from_runtime(snapshot)

    async def qa_memory_view(
        self,
        thread_id: str,
        *,
        current_message: str = "",
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> QAMemoryView:
        """返回 QA 的冻结账号资料和有限近期对话视图。"""
        snapshot = runtime_memory or await self.load_runtime_memory(
            thread_id,
            current_message=current_message,
        )
        self._assert_runtime_belongs_to(thread_id, snapshot)
        return self._qa_memory_view_from_runtime(
            snapshot,
            current_message=current_message,
        )

    def _enqueue_account_promotion(
        self,
        thread_id: str,
        memory: ConversationMemory,
        user_texts: list[str],
        *,
        channel: str | None,
    ) -> asyncio.Task | None:
        """把较慢的 PG/抽取器升级放进独立队列，不阻塞 session 写链。"""
        if not user_texts:
            return None
        previous = self._profile_task_tails.get(thread_id)
        memory_snapshot = deepcopy(memory)

        async def runner() -> None:
            if previous is not None:
                await previous
            for user_text in user_texts:
                try:
                    await self._save_account_user_memory(
                        memory_snapshot,
                        user_text,
                    )
                except Exception as exc:
                    logger.warning(
                        "后台饮食画像写入失败: channel=%s error_type=%s",
                        channel or conversation_channel(thread_id),
                        type(exc).__name__,
                    )
                try:
                    await self.capture_general_profile_facts(
                        thread_id,
                        user_text,
                        passive=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "后台通用画像写入失败: channel=%s error_type=%s",
                        channel or conversation_channel(thread_id),
                        type(exc).__name__,
                    )

        task = asyncio.create_task(runner())
        self._pending_tasks.add(task)
        self._profile_task_tails[thread_id] = task

        def completed(done: asyncio.Task) -> None:
            self._pending_tasks.discard(done)
            if self._profile_task_tails.get(thread_id) is done:
                self._profile_task_tails.pop(thread_id, None)

        task.add_done_callback(completed)
        return task

    def enqueue_turn(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        channel: str | None = None,
        user_id: str | None = None,
        promote_account_memory: bool = True,
    ) -> asyncio.Task | None:
        """按 thread 保序写入 session；用户文本可独立决定是否升级账号画像。"""
        clean = str(content or "").strip()
        if not clean:
            return None
        previous = self._task_tails.get(thread_id)

        async def runner() -> bool:
            if previous is not None:
                await previous
            try:
                memory = await self.append_turn(
                    thread_id,
                    role,
                    clean,
                    channel=channel,
                    user_id=user_id,
                    persist_account_memory=False,
                )
                if role == "user" and promote_account_memory:
                    self._pending_profile_texts.setdefault(thread_id, []).append(clean)
                elif role == "assistant":
                    pending_profile_texts = list(
                        self._pending_profile_texts.get(thread_id, [])
                    )
                    self._pending_profile_texts.pop(thread_id, None)
                    self._enqueue_account_promotion(
                        thread_id,
                        memory,
                        pending_profile_texts,
                        channel=channel,
                    )
                return True
            except Exception as exc:
                logger.warning(
                    "后台会话轮次写入失败: channel=%s role=%s error_type=%s",
                    channel or conversation_channel(thread_id),
                    role,
                    type(exc).__name__,
                )
                return False

        task = asyncio.create_task(runner())
        self._pending_tasks.add(task)
        self._task_tails[thread_id] = task

        def completed(done: asyncio.Task) -> None:
            self._pending_tasks.discard(done)
            if self._task_tails.get(thread_id) is done:
                self._task_tails.pop(thread_id, None)

        task.add_done_callback(completed)
        return task

    async def wait_for_session_writes(
        self,
        thread_id: str,
        *,
        timeout_seconds: float = 1.0,
    ) -> bool:
        """等待该 thread 已排队的轮次落入短期会话，超时则保留后台重试。"""
        tail = self._task_tails.get(thread_id)
        if tail is None or tail is asyncio.current_task():
            return True
        try:
            saved = await asyncio.wait_for(
                asyncio.shield(tail),
                timeout=max(0.05, float(timeout_seconds)),
            )
        except TimeoutError:
            logger.warning(
                "等待会话轮次写入超时，本轮按降级上下文继续: channel=%s",
                conversation_channel(thread_id),
            )
            return False
        return bool(saved)

    async def drain_pending(self, timeout_seconds: float = 2.0) -> None:
        deadline = asyncio.get_running_loop().time() + max(
            0.05,
            float(timeout_seconds),
        )
        try:
            while self._pending_tasks:
                pending = list(self._pending_tasks)
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    await asyncio.gather(*pending, return_exceptions=True)
        except TimeoutError:
            pending = list(self._pending_tasks)
            for task in pending:
                if not task.done():
                    task.cancel()

    async def _drain_thread_tail(
        self,
        thread_id: str,
        *,
        include_profile: bool = False,
    ) -> None:
        """跨过 session 队列；隐私清除时也跨过画像队列。"""
        session_tail = self._task_tails.get(thread_id)
        if session_tail is not None and session_tail is not asyncio.current_task():
            await asyncio.gather(session_tail, return_exceptions=True)
        if not include_profile:
            return
        profile_tail = self._profile_task_tails.get(thread_id)
        if profile_tail is not None and profile_tail is not asyncio.current_task():
            await asyncio.gather(profile_tail, return_exceptions=True)

    async def append_turn(self, thread_id: str, role: str, content: str, *,
                          channel: str | None = None, user_id: str | None = None,
                          persist_account_memory: bool = True) -> ConversationMemory:
        async with self._lock(thread_id):
            memory = await self.load(thread_id, channel=channel, user_id=user_id)
            clean = str(content or "").strip()
            if not clean:
                return memory
            memory.recent_turns.append(ConversationTurn(role=role, content=clean))
            memory.estimated_tokens += max(1, len(clean) // 2)
            if role == "user":
                memory.total_turns += 1
                memory.turns_since_summary += 1
                memory.preferences = update_preferences(memory.preferences, clean)
            # 正常情况下等助手回复后再压缩，避免把同一轮的用户问题与回答拆开。
            # 单条异常超长消息超过双倍预算时仍立即压缩，防止 payload 失控。
            if role == "assistant" or memory.estimated_tokens >= self.max_tokens * 2:
                self._compact_if_needed(memory)
            await self._save(memory)
            if role == "user" and persist_account_memory:
                await self._save_account_user_memory(memory, clean)
            return memory

    async def apply_session_preference_mutations(
        self,
        thread_id: str,
        mutations: list[PreferenceMutation],
        *,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> tuple[ConversationMemory, list[PreferenceMutation]]:
        """在继续路由前把用户明确偏好写入短期会话存储。"""
        if not mutations:
            return (
                await self.load(thread_id, channel=channel, user_id=user_id),
                [],
            )
        async with self._lock(thread_id):
            memory = await self.load(
                thread_id,
                channel=channel,
                user_id=user_id,
            )
            updated, changed = apply_preference_mutations(
                memory.preferences,
                mutations,
            )
            memory.preferences = sanitize_preferences(updated)
            if changed:
                await self._save(memory)
            return memory, changed

    async def save_search(self, thread_id: str, original_question: str, search_query: str,
                          search_result: dict) -> None:
        async with self._lock(thread_id):
            memory = await self.load(thread_id)
            recommendation = search_result.get("_recommendation") or {}
            reasons = recommendation.get("recipe_reasons") or {}
            def serialize_results(results, *, limit: int) -> list[dict]:
                items = []
                for result in (results or [])[:limit]:
                    md = result.get("metadata") or {}
                    recipe_id = str(md.get("recipe_id") or result.get("id") or "")
                    items.append({
                        "id": recipe_id,
                        "name": md.get("name") or "",
                        "image_url": md.get("image_url") or "",
                        "ingredients": md.get("ingredients") or [],
                        "seasonings": md.get("seasonings") or [],
                        "tags": md.get("tags") or [],
                        "facets": md.get("facets") or {},
                        "description": md.get("description") or "",
                        # Demo 多轮链路需要在“换一批 / 上一批 / 详情 N”后继续展示
                        # 同一份菜谱详情。生产版可再拆成 recipe_id 按需回查，避免快照膨胀。
                        "recipe_detail": (
                            md.get("recipe_detail")
                            if isinstance(md.get("recipe_detail"), dict)
                            else None
                        ),
                        "menu_role": result.get("menu_role") or "",
                        "score": result.get("score", 0),
                        # 自由推荐文案不再作为菜谱事实持久化，避免旧缓存串味。
                        "recommendation_reason": "",
                    })
                return items

            recipes = serialize_results(search_result.get("results"), limit=10)
            candidate_pool = serialize_results(
                search_result.get("_candidate_pool") or search_result.get("results"),
                limit=20,
            )
            snapshot = {
                "decision_id": str(search_result.get("_decision_id") or ""),
                "original_question": original_question,
                "search_query": search_query,
                "search_request": search_result.get("_search_request") or {},
                "recipes": recipes,
                "candidate_pool": candidate_pool,
                "recommendation": recommendation,
                "menu_plan": search_result.get("_menu_plan") or {},
                "created_at": time.time(),
            }
            history = list(memory.search_history or [])
            # 兼容只有 latest_search、尚未生成 search_history 的旧快照。
            if not history and memory.latest_search:
                history.append(memory.latest_search)
            history.append(snapshot)
            memory.search_history = history[-6:]
            memory.latest_search = snapshot
            event = self._search_snapshot_event(snapshot)
            memory.food_history = self._append_food_event(memory.food_history, event)
            await self._save(memory)
            await self._save_account_food_event(memory, event)

    async def record_cooked_recipe(self, thread_id: str, recipe_id: str, name: str) -> None:
        """记录用户已确认且设备实际启动成功的菜，不把“点开/选中”误当成做过。"""
        async with self._lock(thread_id):
            memory = await self.load(thread_id)
            recipe = self._find_recipe(memory, recipe_id, name)
            event = {
                "kind": "cooked",
                "recipe": recipe,
                "created_at": time.time(),
            }
            memory.food_history = self._append_food_event(memory.food_history, event)
            await self._save(memory)
            await self._save_account_food_event(memory, event)

    def _separate_temporal_dietary_constraints(
        self,
        preferences: dict[str, list[str]],
        facts: list[dict],
        *,
        fallback_updated_at: float,
    ) -> tuple[dict[str, list[str]], list[dict]]:
        """稳定约束直接使用；阶段性目标只作为待确认事实暴露给路由。"""
        now = time.time()
        cleaned = sanitize_preferences(preferences)
        facts_by_value = {
            str(item.get("value") or "").strip(): item
            for item in facts or []
            if (
                isinstance(item, dict)
                and str(item.get("type") or "") == "dietary_constraint"
                and str(item.get("status") or "active") == "active"
            )
        }
        stable: list[str] = []
        temporal: list[dict] = []
        for value in cleaned.get("dietary_constraints") or []:
            if not is_transient_dietary_constraint(value):
                stable.append(value)
                continue
            fact = facts_by_value.get(value) or {}
            last_confirmed_at = float(
                fact.get("last_confirmed_at")
                or fact.get("updated_at")
                or fact.get("created_at")
                or fallback_updated_at
                or 0
            )
            expires_at = float(
                fact.get("expires_at")
                or (
                    last_confirmed_at + self.transient_diet_ttl_seconds
                    if last_confirmed_at
                    else 0
                )
            )
            if expires_at <= now:
                continue
            temporal.append({
                "value": value,
                "last_confirmed_at": last_confirmed_at,
                "expires_at": expires_at,
                "source": str(fact.get("source") or "user_explicit"),
            })
        cleaned["dietary_constraints"] = stable
        return cleaned, temporal

    async def account_memory_context(
        self,
        thread_id: str,
        *,
        memory: ConversationMemory | None = None,
    ) -> dict:
        """账号长期记忆：偏好、用户摘要与饮食行为；thread 只作为旧数据迁移来源。"""
        memory = memory or await self.load(thread_id)
        profile_key = self._profile_key(memory)
        if profile_key:
            profile = await self._load_profile(profile_key)
            # 已识别账号时，账号画像是唯一事实源。这样用户清除账号记忆后，
            # 不会又被某个旧群聊 thread 的本地副本“复活”。
            events = list(profile.food_history or []) if profile is not None else []
            # 从升级前的 search_history 自动回填账号画像，无需人工迁库。
            # 排序后再去重，避免 save_search 同时写入的新旧两份记录重复。
            cleared_at = float(((profile.summary if profile else {}) or {}).get("food_memory_cleared_at") or 0)
            legacy_events = [
                self._search_snapshot_event(snapshot)
                for snapshot in (memory.search_history or [])
                if snapshot and float(snapshot.get("created_at") or 0) > cleared_at
            ]
            combined = sorted(
                [*events, *legacy_events],
                key=lambda item: float(item.get("created_at") or 0),
            )
            migrated = []
            for event in combined:
                migrated = self._append_food_event(migrated, event)
            events = migrated
            if legacy_events and events != list((profile.food_history if profile else []) or []):
                async with self._lock(profile_key):
                    if profile is None:
                        profile = ConversationMemory(
                            thread_id=profile_key,
                            channel=memory.channel,
                            user_id=memory.user_id,
                        )
                    profile.food_history = events
                    profile.version += 1
                    profile.updated_at = time.time()
                    await self._save_profile(
                        profile,
                        expires_at=profile.updated_at + self.profile_ttl_seconds,
                    )
            profile_preferences = sanitize_preferences((profile.preferences if profile else {}) or {})
            merged_preferences = self._merge_preferences(profile_preferences, memory.preferences)
            merged_preferences, temporal_dietary_constraints = (
                self._separate_temporal_dietary_constraints(
                    merged_preferences,
                    list((profile.long_term_facts if profile else []) or []),
                    fallback_updated_at=float(
                        (profile.updated_at if profile else memory.updated_at)
                        or 0
                    ),
                )
            )
            digest = self._canonical_account_digest(
                merged_preferences,
                temporal_dietary_constraints,
            )
            preferred_name = (
                str((profile.preferred_name if profile else None) or "").strip()
                or None
            )
            display_name = (
                str((profile.display_name if profile else None) or "").strip()
                or None
            )
        else:
            events = list(memory.food_history or [])
            merged_preferences = sanitize_preferences(memory.preferences)
            merged_preferences, temporal_dietary_constraints = (
                self._separate_temporal_dietary_constraints(
                    merged_preferences,
                    list(memory.long_term_facts or []),
                    fallback_updated_at=float(memory.updated_at or 0),
                )
            )
            digest = self._canonical_account_digest(
                merged_preferences,
                temporal_dietary_constraints,
            )
            preferred_name = None
            display_name = None
        return {
            "preferred_name": preferred_name,
            "display_name": display_name,
            "preferences": merged_preferences,
            "temporal_dietary_constraints": temporal_dietary_constraints,
            "account_digest": digest,
            "events": events[-20:],
        }

    async def preference_context(self, thread_id: str) -> dict:
        """精确任务只读账号稳定偏好，不触碰 Redis 短期会话。"""
        channel, user_id = conversation_identity(thread_id)
        shell = ConversationMemory(
            thread_id=thread_id,
            channel=channel,
            user_id=user_id,
        )
        profile_key = self._profile_key(shell)
        if not profile_key:
            return {
                "preferred_name": None,
                "display_name": None,
                "preferences": {},
                "temporal_dietary_constraints": [],
                "account_digest": [],
                "events": [],
            }
        profile = await self._load_profile(profile_key)
        preferences, temporal_dietary_constraints = (
            self._separate_temporal_dietary_constraints(
                sanitize_preferences((profile.preferences if profile else {}) or {}),
                list((profile.long_term_facts if profile else []) or []),
                fallback_updated_at=float((profile.updated_at if profile else 0) or 0),
            )
        )
        return {
            "preferred_name": (
                str((profile.preferred_name if profile else None) or "").strip()
                or None
            ),
            "display_name": (
                str((profile.display_name if profile else None) or "").strip()
                or None
            ),
            "preferences": preferences,
            "temporal_dietary_constraints": temporal_dietary_constraints,
            "account_digest": [],
            "events": [],
        }

    async def routing_context(
        self,
        thread_id: str,
        *,
        scope: MemoryScope = "full",
        current_message: str = "",
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> dict:
        """从单轮快照派生路由视图；精确任务只取账号稳定偏好。"""
        if scope not in {"preferences", "route", "full"}:
            raise ValueError(f"unsupported memory scope: {scope}")
        snapshot = runtime_memory or await self.load_runtime_memory(
            thread_id,
            current_message=current_message,
        )
        self._assert_runtime_belongs_to(thread_id, snapshot)
        if scope == "preferences":
            context = self._preference_context_from_runtime(snapshot)
            safety_view = self._safety_memory_view_from_runtime(snapshot)
            return {
                **context,
                **safety_view.to_router_context_dict(),
                "recent_turns": [],
                "recent_user_turns": [],
                "conversation_summary": {},
                "task_state": {},
                "current_message": str(current_message or "").strip(),
                "runtime_memory_ref": snapshot.context_ref(),
            }
        memory = snapshot.short_term
        account_context = self._account_context_from_runtime(snapshot)
        context = build_conversation_context(
            memory,
            account_context,
            current_message=current_message,
            include_recent=scope == "full",
        ).to_dict()
        previous_search = self._previous_search_snapshot(memory)
        context["previous_candidate_count"] = len(
            (previous_search or {}).get("recipes") or []
        )
        if scope == "route":
            planner_view = self._planner_memory_view_from_runtime(snapshot)
            context.update(planner_view.to_planner_context_dict())
            context["recent_turns"] = []
            context["conversation_summary"] = {}
            context["task_state"] = {}
        else:
            qa_view = self._qa_memory_view_from_runtime(
                snapshot,
                current_message=current_message,
            )
            qa_context = qa_view.to_qa_context_dict()
            context.update({
                key: qa_context[key]
                for key in (
                    "preferred_name",
                    "display_name",
                    "preferences",
                    "temporal_dietary_constraints",
                    "account_digest",
                    "long_term_memory_status",
                    "short_term_memory_status",
                    "memory_context_provided",
                    "profile_version",
                    "is_new_session",
                    "hydrated_at",
                )
            })
            # 兼容门面仍返回 ConversationTurn，但每次创建隔离副本，避免调用方
            # 修改对象后污染共享 RuntimeMemorySnapshot。
            context["recent_turns"] = [
                ConversationTurn(
                    role=item.role,
                    content=item.content,
                    created_at=item.created_at,
                )
                for item in list(snapshot.short_term.recent_turns or [])[-10:]
            ]
            context["conversation_summary"] = deepcopy(
                snapshot.short_term.summary or {}
            )
        context["runtime_memory_ref"] = snapshot.context_ref()
        return context

    async def user_profile_context(
        self,
        thread_id: str,
        *,
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> dict:
        """读取本轮可见资料，同时区分 PostgreSQL 值与会话失败覆盖值。"""
        snapshot = runtime_memory or await self.load_runtime_memory(thread_id)
        if not snapshot.belongs_to(thread_id):
            raise ValueError("runtime memory snapshot belongs to another thread")
        memory = snapshot.short_term
        account = self._account_context_from_runtime(snapshot)
        temporary = (
            dict(memory.task_state.temporary_profile)
            if isinstance(memory.task_state.temporary_profile, dict)
            else {}
        )
        has_override = temporary.get("preferred_name_set") is True
        persistent_name = (
            str(account.get("preferred_name") or "").strip() or None
        )
        visible_name = (
            str(temporary.get("preferred_name") or "").strip() or None
            if has_override
            else persistent_name
        )
        return {
            **account,
            "persistent_preferred_name": persistent_name,
            "preferred_name": visible_name,
            "preferred_name_source": (
                "session_override" if has_override else "postgres"
            ),
            "long_term_memory_status": snapshot.long_term.status,
            "profile_version": snapshot.long_term.profile_version,
        }

    async def update_preferred_name(
        self,
        thread_id: str,
        preferred_name: str | None,
        *,
        expected_current: str | None,
    ) -> dict:
        """同步提交明确称呼；CAS 冲突会重新读取，绝不静默覆盖新值。"""
        channel, user_id = conversation_identity(thread_id)
        shell = ConversationMemory(
            thread_id=thread_id,
            channel=channel,
            user_id=user_id,
        )
        profile_key = self._profile_key(shell)
        if not profile_key:
            raise ValueError("persistent profile identity is unavailable")
        target = str(preferred_name or "").strip() or None
        expected = str(expected_current or "").strip() or None
        last_conflict: ConversationStoreConflict | None = None
        result: dict | None = None
        async with self._lock(profile_key):
            for _attempt in range(3):
                profile = await self._load_profile(profile_key)
                if profile is None:
                    profile = ConversationMemory(
                        thread_id=profile_key,
                        channel=channel,
                        user_id=user_id,
                    )
                current = str(profile.preferred_name or "").strip() or None
                if current != expected:
                    raise PreferredNameConflict(
                        "preferred name changed before confirmation"
                    )
                if current == target:
                    result = {
                        "status": "unchanged",
                        "old_value": current,
                        "new_value": target,
                    }
                    break
                profile.preferred_name = target
                profile.long_term_facts = self._sync_preferred_name_fact(
                    profile.long_term_facts,
                    target,
                    source_thread_id=thread_id,
                    source_channel=channel,
                )
                profile.version += 1
                profile.updated_at = time.time()
                try:
                    await self._save_profile(
                        profile,
                        expires_at=profile.updated_at + self.profile_ttl_seconds,
                    )
                except ConversationStoreConflict as exc:
                    last_conflict = exc
                    continue
                result = {
                    "status": "updated",
                    "old_value": current,
                    "new_value": target,
                }
                break
        if result is None:
            if last_conflict is not None:
                raise last_conflict
            raise ConversationStoreConflict("preferred name could not be saved")
        # profile 锁释放后再清理 thread 覆盖，避免嵌套两类锁形成环路。
        try:
            await self.clear_temporary_profile(thread_id)
        except Exception as exc:
            logger.warning(
                "长期称呼已保存，但清理会话覆盖失败: error_type=%s",
                type(exc).__name__,
            )
        return result

    async def apply_general_profile_facts(
        self,
        thread_id: str,
        candidates: list[ProfileFactCandidate],
    ) -> ProfileFactApplyResult:
        """同步提交已校验的通用画像事实；PG 成功前不报告已记住。"""
        if not candidates:
            return ProfileFactApplyResult(facts=[], changed=False)
        channel, user_id = conversation_identity(thread_id)
        shell = ConversationMemory(
            thread_id=thread_id,
            channel=channel,
            user_id=user_id,
        )
        profile_key = self._profile_key(shell)
        if not profile_key:
            raise ValueError("persistent profile identity is unavailable")
        last_conflict: ConversationStoreConflict | None = None
        persisted: ProfileFactApplyResult | None = None
        async with self._lock(profile_key):
            for _attempt in range(3):
                profile = await self._load_profile(profile_key)
                if profile is None:
                    profile = ConversationMemory(
                        thread_id=profile_key,
                        channel=channel,
                        user_id=user_id,
                    )
                applied = apply_profile_fact_mutations(
                    profile.long_term_facts,
                    candidates,
                    profile_key=profile_key,
                    source_channel=channel,
                    source_thread_id=thread_id,
                )
                if not applied.changed:
                    return applied
                profile.long_term_facts = applied.facts
                profile.version += 1
                profile.updated_at = time.time()
                try:
                    await self._save_profile(
                        profile,
                        expires_at=profile.updated_at + self.profile_ttl_seconds,
                    )
                except ConversationStoreConflict as exc:
                    last_conflict = exc
                    continue
                persisted = applied
                break
        if persisted is not None:
            await self._sync_general_profile_index(profile_key, persisted)
            return persisted
        if last_conflict is not None:
            raise last_conflict
        raise ConversationStoreConflict("general profile facts could not be saved")

    @staticmethod
    def _general_memory_text_digest(text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()

    async def capture_general_profile_facts(
        self,
        thread_id: str,
        user_text: str,
        *,
        force: bool = False,
        passive: bool = False,
        mark_handled: bool = False,
    ):
        """提取并写入通用事实；显式命令可标记，避免助手轮次后重复提取。"""
        extractor = self.profile_fact_extractor
        if extractor is None:
            return None, None
        from app.conversation.profile_rollout import general_profile_memory_enabled

        if not general_profile_memory_enabled(thread_id):
            return None, None
        digest = self._general_memory_text_digest(user_text)
        handled = self._handled_general_memory_digests.get(thread_id, set())
        if passive and digest in handled:
            handled.discard(digest)
            if not handled:
                self._handled_general_memory_digests.pop(thread_id, None)
            return None, None
        # 饮食偏好继续归现有确定性写入器所有；避免同一句在两个画像域重复落库。
        if passive and extract_preference_mutations(user_text):
            return None, None
        try:
            extraction = await extractor.extract(user_text, force=force)
            applied = (
                await self.apply_general_profile_facts(thread_id, extraction.facts)
                if extraction.facts
                else None
            )
            logger.info(
                "通用画像提取完成: status=%s candidate_count=%s accepted_count=%s "
                "changed_count=%s",
                extraction.status,
                extraction.candidate_count,
                len(extraction.facts),
                (
                    len(applied.upserted) + len(applied.deleted_ids)
                    if applied is not None
                    else 0
                ),
            )
            return extraction, applied
        finally:
            if mark_handled:
                remembered = self._handled_general_memory_digests.setdefault(
                    thread_id,
                    set(),
                )
                remembered.add(digest)
                while len(remembered) > 20:
                    remembered.pop()

    async def _sync_general_profile_index(
        self,
        profile_key: str,
        applied: ProfileFactApplyResult,
    ) -> None:
        """PG 成功后尽力同步派生索引；日志只记录数量和异常类型。"""
        index = self.profile_memory_index
        if index is None:
            return
        if applied.deleted_ids:
            try:
                await index.delete(profile_key, applied.deleted_ids)
            except Exception as exc:
                logger.warning(
                    "通用记忆索引删除失败: count=%s error_type=%s",
                    len(applied.deleted_ids),
                    type(exc).__name__,
                )
        if applied.upserted:
            try:
                await index.upsert(profile_key, applied.upserted)
            except Exception as exc:
                logger.warning(
                    "通用记忆索引写入失败: count=%s error_type=%s",
                    len(applied.upserted),
                    type(exc).__name__,
                )

    async def _delete_general_profile_index(
        self,
        profile_key: str,
        fact_ids: list[str],
    ) -> None:
        if self.profile_memory_index is None or not fact_ids:
            return
        try:
            await self.profile_memory_index.delete(profile_key, fact_ids)
        except Exception as exc:
            logger.warning(
                "通用记忆索引清理失败: count=%s error_type=%s",
                len(fact_ids),
                type(exc).__name__,
            )

    async def general_profile_facts(
        self,
        thread_id: str,
        *,
        limit: int = 50,
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> list[dict]:
        """读取 PostgreSQL 当前有效通用事实；不把过期值暴露给上层。"""
        if runtime_memory is not None:
            if not runtime_memory.belongs_to(thread_id):
                raise ValueError("runtime memory snapshot belongs to another thread")
            return runtime_memory.long_term.copy_general_facts(limit=limit)
        channel, user_id = conversation_identity(thread_id)
        shell = ConversationMemory(
            thread_id=thread_id,
            channel=channel,
            user_id=user_id,
        )
        profile_key = self._profile_key(shell)
        if not profile_key:
            return []
        profile = await self._load_profile(profile_key)
        return active_general_profile_facts(
            list((profile.long_term_facts if profile else []) or []),
            limit=limit,
        )

    async def recall_general_profile_facts(
        self,
        thread_id: str,
        query: str,
        *,
        limit: int | None = None,
        runtime_memory: RuntimeMemorySnapshot | None = None,
    ) -> list[dict]:
        """Milvus 召回后按 PG 当前事实求交；索引异常时退回最近 PG 事实。"""
        channel, user_id = conversation_identity(thread_id)
        shell = ConversationMemory(
            thread_id=thread_id,
            channel=channel,
            user_id=user_id,
        )
        profile_key = self._profile_key(shell)
        if not profile_key:
            return []
        top_k = max(1, int(limit or self.memory_recall_top_k))
        if runtime_memory is not None:
            if not runtime_memory.belongs_to(thread_id):
                raise ValueError("runtime memory snapshot belongs to another thread")
            active = runtime_memory.long_term.copy_general_facts(limit=50)
        else:
            profile = await self._load_profile(profile_key)
            active = active_general_profile_facts(
                list((profile.long_term_facts if profile else []) or []),
                limit=50,
            )
        if not active:
            return []
        # Milvus 是可重建派生索引。未配置或暂时不可用时，仍可使用本轮已从
        # PostgreSQL 加载的有限当前事实，不能把“没有索引”误报成“没有记忆”。
        if self.profile_memory_index is None:
            return active[:top_k]
        facts_by_id = {
            str(item.get("id") or ""): item
            for item in active
            if str(item.get("id") or "")
        }
        try:
            hits = await self.profile_memory_index.search(
                profile_key,
                query,
                limit=max(top_k * 2, top_k),
            )
        except Exception as exc:
            logger.warning(
                "通用记忆索引召回失败，使用 PG 降级: fact_count=%s error_type=%s",
                len(active),
                type(exc).__name__,
            )
            return active[:top_k]
        selected: list[dict] = []
        seen: set[str] = set()
        for hit in hits:
            fact_id = str(getattr(hit, "fact_id", "") or "")
            score = float(getattr(hit, "score", -1.0))
            if (
                fact_id in seen
                or fact_id not in facts_by_id
                or score < self.memory_recall_min_score
            ):
                continue
            selected.append(dict(facts_by_id[fact_id]))
            seen.add(fact_id)
            if len(selected) >= top_k:
                break
        return selected

    async def clear_general_profile_facts(self, thread_id: str) -> list[str]:
        """物理移除通用事实值；称呼、饮食偏好和行为历史保持不变。"""
        channel, user_id = conversation_identity(thread_id)
        shell = ConversationMemory(
            thread_id=thread_id,
            channel=channel,
            user_id=user_id,
        )
        profile_key = self._profile_key(shell)
        if not profile_key:
            return []
        last_conflict: ConversationStoreConflict | None = None
        persisted_removed_ids: list[str] | None = None
        async with self._lock(profile_key):
            for _attempt in range(3):
                profile = await self._load_profile(profile_key)
                if profile is None:
                    return []
                removed_ids = [
                    str(item.get("id") or "")
                    for item in profile.long_term_facts
                    if (
                        isinstance(item, dict)
                        and str(item.get("type") or "") == "profile_fact"
                        and str(item.get("id") or "")
                    )
                ]
                if not removed_ids:
                    return []
                profile.long_term_facts = [
                    item
                    for item in profile.long_term_facts
                    if not (
                        isinstance(item, dict)
                        and str(item.get("type") or "") == "profile_fact"
                    )
                ]
                profile.version += 1
                profile.updated_at = time.time()
                try:
                    await self._save_profile(
                        profile,
                        expires_at=profile.updated_at + self.profile_ttl_seconds,
                    )
                except ConversationStoreConflict as exc:
                    last_conflict = exc
                    continue
                persisted_removed_ids = removed_ids
                break
        if persisted_removed_ids is not None:
            await self._delete_general_profile_index(
                profile_key,
                persisted_removed_ids,
            )
            return persisted_removed_ids
        if last_conflict is not None:
            raise last_conflict
        raise ConversationStoreConflict("general profile facts could not be cleared")

    async def load_pending_profile_update(
        self,
        thread_id: str,
        *,
        max_age_seconds: int = 300,
        short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
    ) -> dict | None:
        if short_term_snapshot is None:
            task_state = await self.load_task_state(thread_id)
        else:
            if not short_term_snapshot.belongs_to(thread_id):
                raise ValueError(
                    "short-term runtime snapshot belongs to another thread"
                )
            if short_term_snapshot.task_state_status in {
                "unavailable",
                "invalid",
            }:
                raise ConversationBackendUnavailable(
                    "short-term memory is not trustworthy"
                )
            task_state = short_term_snapshot.memory.task_state
        pending = task_state.pending_profile_update
        if not isinstance(pending, dict):
            return None
        expires_at = float(pending.get("expires_at") or 0)
        created_at = float(pending.get("created_at") or 0)
        valid = (
            expires_at > time.time()
            if expires_at
            else time.time() - created_at <= max(1, max_age_seconds)
        )
        if valid:
            return dict(pending)
        # 读路径只做语义过期，不物理删除。否则读到旧值后再重读清理，
        # 会删掉期间并发写入的新 pending；整个 task key 由 TTL 回收。
        return None

    async def save_pending_profile_update(
        self,
        thread_id: str,
        pending: dict,
    ) -> None:
        baseline = await self.load_task_state_record(thread_id)
        before = baseline.state
        after = ConversationTaskState.from_dict(before.to_dict())
        after.pending_profile_update = dict(pending)
        await self.patch_task_state(
            thread_id,
            before,
            after,
            expected_revision=baseline.revision,
            expected_generation=baseline.generation,
        )

    async def clear_pending_profile_update(self, thread_id: str) -> None:
        baseline = await self.load_task_state_record(thread_id)
        before = baseline.state
        if before.pending_profile_update is None:
            return
        after = ConversationTaskState.from_dict(before.to_dict())
        after.pending_profile_update = None
        await self.patch_task_state(
            thread_id,
            before,
            after,
            expected_revision=baseline.revision,
            expected_generation=baseline.generation,
        )

    async def save_temporary_preferred_name(
        self,
        thread_id: str,
        preferred_name: str | None,
    ) -> None:
        """画像写失败时让本会话先按新称呼继续，不伪装成长期保存成功。"""
        baseline = await self.load_task_state_record(thread_id)
        before = baseline.state
        after = ConversationTaskState.from_dict(before.to_dict())
        after.temporary_profile = {
            "preferred_name_set": True,
            "preferred_name": str(preferred_name or "").strip() or None,
            "updated_at": time.time(),
        }
        await self.patch_task_state(
            thread_id,
            before,
            after,
            expected_revision=baseline.revision,
            expected_generation=baseline.generation,
        )

    async def clear_temporary_profile(self, thread_id: str) -> None:
        baseline = await self.load_task_state_record(thread_id)
        before = baseline.state
        if not before.temporary_profile:
            return
        after = ConversationTaskState.from_dict(before.to_dict())
        after.temporary_profile = {}
        await self.patch_task_state(
            thread_id,
            before,
            after,
            expected_revision=baseline.revision,
            expected_generation=baseline.generation,
        )

    async def load_task_state(
        self,
        thread_id: str,
        *,
        short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
    ) -> ConversationTaskState:
        """读取独立任务状态，并返回隔离副本。"""
        record = await self.load_task_state_record(
            thread_id,
            short_term_snapshot=short_term_snapshot,
        )
        return ConversationTaskState.from_dict(record.state.to_dict())

    async def load_task_state_record(
        self,
        thread_id: str,
        *,
        short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
    ) -> TaskStateRecord:
        """读取独立 revision 的任务状态；旧嵌套字段仅作为迁移种子。"""
        snapshot = short_term_snapshot or await self.load_short_term_runtime(thread_id)
        if not snapshot.belongs_to(thread_id):
            raise ValueError("short-term runtime snapshot belongs to another thread")
        if snapshot.task_state_status in {"unavailable", "invalid"}:
            raise ConversationBackendUnavailable("task state is not trustworthy")
        return TaskStateRecord(
            thread_id=str(thread_id),
            state=ConversationTaskState.from_dict(
                snapshot.memory.task_state.to_dict()
            ),
            revision=max(0, int(snapshot.task_state_revision)),
            generation=max(0, int(snapshot.task_state_generation)),
            updated_at=float(snapshot.memory.task_state.updated_at or 0),
        )

    async def commit_task_state(
        self,
        thread_id: str,
        state: ConversationTaskState | dict,
        *,
        expected_revision: int,
        expected_generation: int = 0,
        new_generation: int | None = None,
    ) -> TaskStateRecord:
        """严格 CAS 提交任务态；冲突直接暴露给应用服务，不静默覆盖。"""
        incoming = (
            state
            if isinstance(state, ConversationTaskState)
            else ConversationTaskState.from_dict(state)
        )
        incoming = ConversationTaskState.from_dict(incoming.to_dict())
        self._refresh_task_state_derived(incoming)
        commit = getattr(self.store, "commit_task_state_record", None)
        if commit is None:
            raise ConversationBackendUnavailable(
                "conversation store does not support task state CAS"
            )
        record = TaskStateRecord(
            thread_id=str(thread_id),
            state=incoming,
            revision=max(0, int(expected_revision)),
            generation=(
                max(0, int(expected_generation))
                if new_generation is None
                else max(0, int(new_generation))
            ),
            updated_at=time.time(),
        )
        return await self._call_backend(
            "conversation",
            "commit_task_state_record",
            lambda: commit(
                record,
                expected_revision=max(0, int(expected_revision)),
                expected_generation=max(0, int(expected_generation)),
                new_generation=(
                    None
                    if new_generation is None
                    else max(0, int(new_generation))
                ),
                expires_at=time.time() + self.ttl_seconds,
            ),
            timeout_seconds=self.write_timeout_seconds,
        )

    async def save_task_state(
        self,
        thread_id: str,
        state: ConversationTaskState | dict,
        *,
        expected_revision: int | None = None,
        expected_generation: int | None = None,
    ) -> ConversationTaskState:
        """严格替换整份任务状态；仅允许无版本参数的初始空 key 写入。"""
        incoming = (
            state
            if isinstance(state, ConversationTaskState)
            else ConversationTaskState.from_dict(state)
        )
        incoming = ConversationTaskState.from_dict(incoming.to_dict())
        if (expected_revision is None) != (expected_generation is None):
            raise ValueError(
                "expected_revision and expected_generation must be provided together"
            )
        if expected_revision is None:
            baseline = await self.load_task_state_record(thread_id)
            if baseline.revision != 0 or baseline.generation != 0:
                raise TaskStateStoreConflict(
                    "whole task state replacement requires an explicit baseline"
                )
            expected_revision = baseline.revision
            expected_generation = baseline.generation
        committed = await self.commit_task_state(
            thread_id,
            incoming,
            expected_revision=max(0, int(expected_revision)),
            expected_generation=max(0, int(expected_generation)),
        )
        return ConversationTaskState.from_dict(committed.state.to_dict())

    @staticmethod
    def _refresh_task_state_derived(state: ConversationTaskState) -> None:
        refresh_task_state_derived(state, now=time.time())

    async def patch_task_state(
        self,
        thread_id: str,
        before: ConversationTaskState | dict,
        after: ConversationTaskState | dict,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ConversationTaskState:
        """以调用方读取到的 revision/generation 为起点执行三方合并。

        generation 变化意味着期间发生过 reset，旧请求必须拒绝回写；同一字段
        同时被其他请求修改时也拒绝后写覆盖。
        """
        baseline_state = (
            before
            if isinstance(before, ConversationTaskState)
            else ConversationTaskState.from_dict(before)
        )
        incoming_state = (
            after
            if isinstance(after, ConversationTaskState)
            else ConversationTaskState.from_dict(after)
        )
        baseline = baseline_state.to_dict()
        incoming = incoming_state.to_dict()
        derived_fields = {
            "schema_version",
            "current_task",
            "selected_device_id",
            "language",
            "updated_at",
        }
        changes = {
            key: value
            for key, value in incoming.items()
            if key not in derived_fields and baseline.get(key) != value
        }
        if not changes:
            return await self.load_task_state(thread_id)

        candidate = ConversationTaskState.from_dict(baseline)
        for key, value in changes.items():
            setattr(candidate, key, value)
        revision = max(0, int(expected_revision))
        generation = max(0, int(expected_generation))
        last_conflict: TaskStateStoreConflict | None = None
        for attempt in range(3):
            try:
                committed = await self.commit_task_state(
                    thread_id,
                    candidate,
                    expected_revision=revision,
                    expected_generation=generation,
                )
                return ConversationTaskState.from_dict(committed.state.to_dict())
            except TaskStateStoreConflict as exc:
                last_conflict = exc
                if attempt >= 2:
                    break
                latest = await self.load_task_state_record(thread_id)
                if int(latest.generation) != int(expected_generation):
                    raise TaskStateStoreConflict(
                        "task state generation changed; stale patch cannot cross reset"
                    ) from exc
                latest_values = latest.state.to_dict()
                conflicting = [
                    key
                    for key, desired in changes.items()
                    if baseline.get(key) != latest_values.get(key)
                    and desired != latest_values.get(key)
                ]
                if conflicting:
                    raise TaskStateStoreConflict(
                        "task state field conflict: "
                        + ",".join(sorted(conflicting))
                    ) from exc
                candidate = ConversationTaskState.from_dict(latest_values)
                for key, value in changes.items():
                    setattr(candidate, key, value)
                revision = latest.revision
                generation = latest.generation
        if last_conflict is not None:
            raise last_conflict
        raise TaskStateStoreConflict("task state patch could not be saved")

    async def consume_pending_device_start(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
    ) -> dict | None:
        """跨 worker 原子领取设备启动动作；领取失败时绝不发送设备命令。"""
        result = await self.consume_pending_device_start_with_revision(
            thread_id,
            expected_action_id=expected_action_id,
        )
        return dict(result.payload) if result.claimed else None

    async def consume_pending_device_start_with_revision(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
        max_age_seconds: int = 300,
    ) -> DeviceStartClaim:
        """在独立任务 key 中原子领取，并返回推进后的 revision。"""
        action_id = str(expected_action_id or "").strip()
        if not action_id:
            return DeviceStartClaim("missing", None, 0)
        baseline = await self.load_task_state_record(thread_id)
        atomic_claim = getattr(
            self.store,
            "claim_pending_device_start_record",
            None,
        )
        if atomic_claim is None:
            raise ConversationBackendUnavailable(
                "conversation store does not support atomic task claim"
            )
        # 旧嵌套状态首次被确认前先迁入独立 key。两个 worker 同时迁移时，
        # 只允许一个 revision=0 提交成功，另一个随后直接读取新 key。
        if baseline.revision == 0 and self._task_state_has_data(baseline.state):
            try:
                baseline = await self.commit_task_state(
                    thread_id,
                    baseline.state,
                    expected_revision=0,
                    expected_generation=baseline.generation,
                )
            except TaskStateStoreConflict:
                baseline = await self.load_task_state_record(thread_id)
        return await self._call_backend(
            "conversation",
            "claim_pending_device_start_record",
            lambda: atomic_claim(
                thread_id,
                expected_action_id=action_id,
                max_age_seconds=max(1, int(max_age_seconds)),
                now=time.time(),
            ),
            timeout_seconds=self.write_timeout_seconds,
        )

    async def load_search_clarification(
        self,
        thread_id: str,
        *,
        max_age_seconds: int = 600,
    ) -> dict | None:
        """读取与短期会话同生命周期的结构化澄清状态。"""
        task_state = await self.load_task_state(thread_id)
        state = task_state.pending_search_clarification
        if not isinstance(state, dict):
            return None
        if time.time() - float(state.get("ts") or 0) <= max(1, max_age_seconds):
            return dict(state)
        # 只返回语义过期，读路径不删共享状态，避免误删并发新值。
        return None

    async def save_search_clarification(self, thread_id: str, state: dict) -> None:
        """原子保存澄清事实，避免仅依赖单进程全局字典。"""
        if not isinstance(state, dict) or not isinstance(state.get("request"), dict):
            return
        baseline = await self.load_task_state_record(thread_id)
        before = baseline.state
        after = ConversationTaskState.from_dict(before.to_dict())
        after.pending_search_clarification = dict(state)
        await self.patch_task_state(
            thread_id,
            before,
            after,
            expected_revision=baseline.revision,
            expected_generation=baseline.generation,
        )

    async def clear_search_clarification(self, thread_id: str) -> None:
        """只清除待澄清任务，不影响已保存的会话轮次与搜索历史。"""
        baseline = await self.load_task_state_record(thread_id)
        before = baseline.state
        if before.pending_search_clarification is None:
            return
        after = ConversationTaskState.from_dict(before.to_dict())
        after.pending_search_clarification = None
        await self.patch_task_state(
            thread_id,
            before,
            after,
            expected_revision=baseline.revision,
            expected_generation=baseline.generation,
        )

    async def food_memory_context(self, thread_id: str) -> dict:
        """兼容旧调用；新代码优先使用 account_memory_context。"""
        return await self.account_memory_context(thread_id)

    @staticmethod
    def _previous_search_snapshot(memory: ConversationMemory) -> dict | None:
        """只读取得当前搜索游标的上一批，不移动游标。"""
        history = list(memory.search_history or [])
        if not history and memory.latest_search:
            history = [memory.latest_search]
        if len(history) < 2:
            return None
        current_at = (memory.latest_search or {}).get("created_at")
        current_index = next(
            (
                index
                for index, item in enumerate(history)
                if item.get("created_at") == current_at
            ),
            len(history) - 1,
        )
        if current_index <= 0:
            return None
        return history[current_index - 1]

    async def activate_previous_search(self, thread_id: str) -> dict | None:
        """把当前讨论对象切到上一轮搜索，不发起新检索。"""
        async with self._lock(thread_id):
            memory = await self.load(thread_id)
            history = list(memory.search_history or [])
            if not history and memory.latest_search:
                history = [memory.latest_search]
                memory.search_history = history
            if len(history) < 2:
                return None

            previous = self._previous_search_snapshot(memory)
            if previous is None:
                return None
            memory.latest_search = previous
            await self._save(memory)
            return previous

    async def reset_session(self, thread_id: str) -> ConversationTaskState:
        """只重置当前短期会话，保留账号画像和已经启动的设备任务。

        ``clear`` 是用户明确删除长期记忆时使用的隐私接口；“新对话/重新开始”
        不能复用它。活动设备任务已真实下发，清聊天不能把它伪装成已停止，因此
        在新的短期会话快照中只保留 ``active_cooking`` 供后续查询或明确停止。
        """
        await self._drain_thread_tail(thread_id)
        async with self._lock(thread_id):
            _memory, committed = await self._reset_short_term_state(thread_id)
            self._pending_profile_texts.pop(thread_id, None)
            self._handled_general_memory_digests.pop(thread_id, None)
        return ConversationTaskState.from_dict(committed.state.to_dict())

    @staticmethod
    def _preserve_device_execution(
        state: ConversationTaskState,
    ) -> ConversationTaskState:
        preserved = ConversationTaskState()
        if isinstance(state.active_cooking, dict):
            preserved.active_cooking = dict(state.active_cooking)
        if isinstance(state.device_execution, dict):
            preserved.device_execution = dict(state.device_execution)
        if preserved.active_cooking or preserved.device_execution:
            ConversationService._refresh_task_state_derived(preserved)
        return preserved

    async def _reset_short_term_state(
        self,
        thread_id: str,
    ) -> tuple[ConversationMemory, TaskStateRecord]:
        """严格原子推进 conversation/task 两个 generation，冲突时整体重试。"""
        atomic_reset = getattr(self.store, "reset_with_task_state", None)
        if atomic_reset is None:
            raise ConversationBackendUnavailable(
                "conversation store does not support atomic short-term reset"
            )
        last_conflict: Exception | None = None
        for _attempt in range(3):
            memory = await self.load(thread_id)
            task_baseline = await self.load_task_state_record(thread_id)
            preserved = self._preserve_device_execution(task_baseline.state)
            # 空状态也重算派生字段，保证兼容镜像与独立 key 完全一致。
            self._refresh_task_state_derived(preserved)
            fresh = ConversationMemory(
                thread_id=thread_id,
                channel=memory.channel,
                user_id=memory.user_id,
                task_state_storage_version=2,
                task_state=ConversationTaskState.from_dict(
                    preserved.to_dict()
                ),
            )
            try:
                committed = await self._call_backend(
                    "conversation",
                    "reset_with_task_state",
                    lambda: atomic_reset(
                        fresh,
                        preserved,
                        expected_version=memory.version,
                        expected_generation=memory.conversation_generation,
                        expected_task_revision=task_baseline.revision,
                        expected_task_generation=task_baseline.generation,
                        expires_at=time.time() + self.ttl_seconds,
                    ),
                    timeout_seconds=self.write_timeout_seconds,
                )
            except (ConversationStoreConflict, TaskStateStoreConflict) as exc:
                last_conflict = exc
                continue
            return memory, committed
        if last_conflict is not None:
            raise last_conflict
        raise TaskStateStoreConflict("short-term state reset could not be committed")

    async def remove_preferences(self, thread_id: str, values: list[str]) -> list[str]:
        """响应用户纠错，同时删除当前 thread 与账号长期画像中的偏好。"""
        targets = {str(value).strip() for value in values if str(value).strip()}
        if not targets:
            return []
        async with self._lock(thread_id):
            memory = await self.load(thread_id)
            removed = []
            for key, items in memory.preferences.items():
                kept = []
                for item in items:
                    if item in targets:
                        removed.append(item)
                    else:
                        kept.append(item)
                memory.preferences[key] = kept
            summary_preferences = (memory.summary or {}).get("preferences") or {}
            for key, items in summary_preferences.items():
                summary_preferences[key] = [item for item in items if item not in targets]
            if removed:
                await self._save(memory)
            profile_key = self._profile_key(memory)
            if profile_key:
                async with self._lock(profile_key):
                    profile = await self._load_profile(profile_key)
                    if profile is not None:
                        for key, items in profile.preferences.items():
                            profile.preferences[key] = [item for item in items if item not in targets]
                        summary_preferences = (profile.summary or {}).get("preferences") or {}
                        for key, items in summary_preferences.items():
                            summary_preferences[key] = [item for item in items if item not in targets]
                        profile.long_term_facts = self._sync_long_term_facts(
                            profile.long_term_facts,
                            profile.preferences,
                            source_thread_id=thread_id,
                            source_channel=memory.channel,
                        )
                        stable_preferences, temporal = (
                            self._separate_temporal_dietary_constraints(
                                profile.preferences,
                                profile.long_term_facts,
                                fallback_updated_at=float(profile.updated_at or time.time()),
                            )
                        )
                        profile.summary["account_digest"] = (
                            self._canonical_account_digest(
                                stable_preferences,
                                temporal,
                            )
                        )
                        profile.version += 1
                        profile.updated_at = time.time()
                        await self._save_profile(
                            profile,
                            expires_at=profile.updated_at + self.profile_ttl_seconds,
                        )
            return list(dict.fromkeys(removed))

    async def clear(self, thread_id: str) -> ConversationTaskState:
        # 通道消息采用 per-thread 异步队列写入。先等清除命令之前已经入队的
        # user/assistant 写链结束，再落清除标记；否则旧 user text 可能在清除
        # 返回后才被 assistant tail 写回 PostgreSQL，复活已删除资料。
        await self._drain_thread_tail(thread_id, include_profile=True)
        profile_key: str | None = None
        removed_general_ids: list[str] = []
        async with self._lock(thread_id):
            # 先原子清除可执行的 pending/action，再处理 PostgreSQL 画像。
            # 后续画像删除即使失败，也不会留下仍可领取的旧设备动作。
            memory, committed = await self._reset_short_term_state(thread_id)
            profile_key = self._profile_key(memory)
            if profile_key:
                # 保留一个不含饮食数据的清除标记，防止其他旧 thread 的
                # search_history 在下一次读取时把用户主动删除的账号记忆重新回填。
                cleared = await self._load_profile(profile_key)
                if cleared is None:
                    cleared = ConversationMemory(
                        thread_id=profile_key,
                        channel=memory.channel,
                        user_id=memory.user_id,
                    )
                removed_general_ids = [
                    str(item.get("id") or "")
                    for item in cleared.long_term_facts
                    if (
                        isinstance(item, dict)
                        and str(item.get("type") or "") == "profile_fact"
                        and str(item.get("id") or "")
                    )
                ]
                cleared.preferences = sanitize_preferences({})
                cleared.preferred_name = None
                cleared.food_history = []
                # 用户明确“清除记忆”时物理移除事实值。仅保留不含个人内容的时间
                # 标记，防止其它旧 Redis thread 把历史搜索重新回填到账号画像。
                cleared.long_term_facts = []
                cleared.summary = {"food_memory_cleared_at": time.time()}
                cleared.version += 1
                cleared.updated_at = time.time()
                await self._save_profile(
                    cleared,
                    expires_at=cleared.updated_at + self.profile_ttl_seconds,
                )
            self._pending_profile_texts.pop(thread_id, None)
            self._handled_general_memory_digests.pop(thread_id, None)
        if profile_key and removed_general_ids:
            await self._delete_general_profile_index(
                profile_key,
                removed_general_ids,
            )
        return ConversationTaskState.from_dict(committed.state.to_dict())

    @staticmethod
    def _append_food_event(history: list[dict], event: dict, limit: int = 40) -> list[dict]:
        items = list(history or [])
        signature = (
            event.get("kind"),
            event.get("query"),
            str((event.get("recipe") or {}).get("id") or ""),
        )
        if items:
            previous = items[-1]
            previous_signature = (
                previous.get("kind"),
                previous.get("query"),
                str((previous.get("recipe") or {}).get("id") or ""),
            )
            if signature == previous_signature:
                return items
        items.append(event)
        return items[-limit:]

    @staticmethod
    def _find_recipe(memory: ConversationMemory, recipe_id: str, name: str) -> dict:
        wanted_id = str(recipe_id or "")
        snapshots = [memory.latest_search, *(reversed(memory.search_history or []))]
        for snapshot in snapshots:
            for recipe in (snapshot or {}).get("recipes") or []:
                if wanted_id and str(recipe.get("id") or "") == wanted_id:
                    return dict(recipe)
                if name and recipe.get("name") == name:
                    return dict(recipe)
        return {"id": wanted_id, "name": str(name or "")}

    @staticmethod
    def _search_snapshot_event(snapshot: dict) -> dict:
        return {
            "kind": "searched",
            "query": str(snapshot.get("search_query") or ""),
            "original_question": str(snapshot.get("original_question") or ""),
            "search_request": dict(snapshot.get("search_request") or {}),
            "recipes": list(snapshot.get("recipes") or [])[:5],
            "created_at": float(snapshot.get("created_at") or time.time()),
        }

    @staticmethod
    def _profile_key(memory: ConversationMemory) -> str | None:
        identity = _parse_account_identity(memory.thread_id)
        if identity is None:
            return None
        channel, account_id, user_id = identity
        if memory.channel != channel or memory.user_id != user_id:
            return None
        if channel == "weixin":
            return f"profile:weixin:{account_id}:{user_id}"
        return f"profile:{channel}:{user_id}"

    async def _save_account_food_event(self, memory: ConversationMemory, event: dict) -> None:
        profile_key = self._profile_key(memory)
        if not profile_key:
            return
        async with self._lock(profile_key):
            profile = await self._load_profile(profile_key)
            if profile is None:
                profile = ConversationMemory(
                    thread_id=profile_key,
                    channel=memory.channel,
                    user_id=memory.user_id,
                )
            profile.food_history = self._append_food_event(profile.food_history, event)
            profile.version += 1
            profile.updated_at = time.time()
            await self._save_profile(
                profile,
                expires_at=profile.updated_at + self.profile_ttl_seconds,
            )

    async def _save_account_user_memory(self, memory: ConversationMemory, user_text: str) -> None:
        """只有本轮明确声明的稳定偏好才进入账号长期画像。"""
        profile_key = self._profile_key(memory)
        if not profile_key:
            return
        mutations = [
            mutation
            for mutation in extract_preference_mutations(user_text)
            if (
                mutation.scope == "long_term_candidate"
                or (
                    mutation.scope == "temporal"
                    and mutation.bucket == "dietary_constraints"
                )
            )
        ]
        if not mutations:
            return
        refreshed_preferences, _ = apply_preference_mutations(
            {},
            [
                mutation
                for mutation in mutations
                if mutation.operation == "add"
            ],
        )
        refreshed_preferences = sanitize_preferences(refreshed_preferences)
        async with self._lock(profile_key):
            profile = await self._load_profile(profile_key)
            if profile is None:
                profile = ConversationMemory(
                    thread_id=profile_key,
                    channel=memory.channel,
                    user_id=memory.user_id,
                )
            profile.preferences, _changed = apply_preference_mutations(
                profile.preferences,
                mutations,
            )
            profile.preferences = sanitize_preferences(profile.preferences)
            # “家里现在有什么”只属于当前会话，旧版本误写入画像的值也在
            # 下一次账号画像更新时清空，避免跨天推荐仍使用过期食材。
            profile.preferences["available_ingredients"] = []
            profile.long_term_facts = self._sync_long_term_facts(
                profile.long_term_facts,
                profile.preferences,
                source_thread_id=memory.thread_id,
                source_channel=memory.channel,
                refreshed_preferences=refreshed_preferences,
                transient_diet_ttl_seconds=self.transient_diet_ttl_seconds,
            )
            stable_preferences, temporal = self._separate_temporal_dietary_constraints(
                profile.preferences,
                profile.long_term_facts,
                fallback_updated_at=float(profile.updated_at or time.time()),
            )
            # 旧实现保存整句用户原文；改为从结构化当前态重新生成，避免把
            # 历史请求或提示注入片段再次送给回复模型。
            profile.summary["account_digest"] = self._canonical_account_digest(
                stable_preferences,
                temporal,
            )
            profile.version += 1
            profile.updated_at = time.time()
            await self._save_profile(
                profile,
                expires_at=profile.updated_at + self.profile_ttl_seconds,
            )

    @staticmethod
    def _sync_long_term_facts(
        facts: list[dict],
        preferences: dict[str, list[str]],
        *,
        source_thread_id: str,
        source_channel: str,
        refreshed_preferences: dict[str, list[str]] | None = None,
        transient_diet_ttl_seconds: int = 2_592_000,
    ) -> list[dict]:
        """将当前有效偏好同步为有来源、可删除的长期事实。"""
        now = time.time()
        type_by_key = {
            "likes": "preference_like",
            "dislikes": "preference_dislike",
            "allergens": "allergy",
            "dietary_constraints": "dietary_constraint",
        }
        desired = {
            (type_by_key[key], value)
            for key, values in sanitize_preferences(preferences).items()
            if key in type_by_key
            for value in values
        }
        refreshed = {
            (type_by_key[key], value)
            for key, values in sanitize_preferences(refreshed_preferences or {}).items()
            if key in type_by_key
            for value in values
        }
        existing = [dict(item) for item in (facts or []) if isinstance(item, dict)]
        by_signature = {
            (str(item.get("type") or ""), str(item.get("value") or "")): item
            for item in existing
        }
        for fact_type, value in desired:
            item = by_signature.get((fact_type, value))
            if item is None:
                fingerprint = hashlib.sha256(
                    f"{fact_type}\0{value}".encode("utf-8")
                ).hexdigest()
                item = {
                    "id": f"mem_{fingerprint[:24]}",
                    "type": fact_type,
                    "key": fact_type,
                    "value": value,
                    "source": "user_explicit",
                    "source_channel": source_channel,
                    "source_thread_id": source_thread_id,
                    "confidence": 1.0,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                }
                existing.append(item)
                by_signature[(fact_type, value)] = item
            elif item.get("status") != "active":
                item["status"] = "active"
                item.pop("deleted_at", None)
                item["source_channel"] = source_channel
                item["source_thread_id"] = source_thread_id
                item["updated_at"] = now
            if (
                fact_type == "dietary_constraint"
                and is_transient_dietary_constraint(value)
                and ((fact_type, value) in refreshed or "expires_at" not in item)
            ):
                item["last_confirmed_at"] = now
                item["expires_at"] = now + max(
                    1,
                    int(transient_diet_ttl_seconds),
                )
        for signature, item in by_signature.items():
            if signature[0] not in type_by_key.values() or signature in desired:
                continue
            if item.get("status") == "active":
                item["status"] = "deleted"
                item["deleted_at"] = now
                item["updated_at"] = now
        return existing[-100:]

    @staticmethod
    def _sync_preferred_name_fact(
        facts: list[dict],
        preferred_name: str | None,
        *,
        source_thread_id: str,
        source_channel: str,
    ) -> list[dict]:
        """更新称呼事实历史；当前有效值仍只读取 preferred_name 列。"""
        now = time.time()
        target = str(preferred_name or "").strip()
        existing = [
            dict(item)
            for item in (facts or [])
            if isinstance(item, dict)
        ]
        active_target = None
        for item in existing:
            if str(item.get("type") or "") != "preferred_name":
                continue
            value = str(item.get("value") or "").strip()
            if target and value == target:
                active_target = item
                continue
            if item.get("status") == "active":
                item["status"] = "deleted"
                item["deleted_at"] = now
                item["updated_at"] = now
        if target:
            if active_target is None:
                fingerprint = hashlib.sha256(
                    f"preferred_name\0{target}".encode("utf-8")
                ).hexdigest()
                active_target = {
                    "id": f"mem_{fingerprint[:24]}",
                    "type": "preferred_name",
                    "key": "preferred_name",
                    "value": target,
                    "source": "user_explicit",
                    "confidence": 1.0,
                    "created_at": now,
                }
                existing.append(active_target)
            active_target.update({
                "status": "active",
                "source_channel": source_channel,
                "source_thread_id": source_thread_id,
                "updated_at": now,
            })
            active_target.pop("deleted_at", None)
        return existing[-100:]

    @staticmethod
    def _mark_facts_deleted(facts: list[dict]) -> list[dict]:
        now = time.time()
        result = []
        for raw in facts or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if item.get("status") == "active":
                item["status"] = "deleted"
                item["deleted_at"] = now
                item["updated_at"] = now
            result.append(item)
        return result[-100:]

    @staticmethod
    def _canonical_account_digest(
        preferences: dict[str, list[str]],
        temporal_dietary_constraints: list[dict],
    ) -> list[str]:
        labels = {
            "likes": "饮食偏好-喜欢",
            "dislikes": "饮食偏好-不喜欢或排除",
            "allergens": "饮食安全-过敏原",
            "dietary_constraints": "长期饮食要求",
        }
        cleaned = sanitize_preferences(preferences or {})
        lines = [
            f"{labels[key]}：{'、'.join(values[:20])}"
            for key, values in cleaned.items()
            if key in labels and values
        ]
        temporal = [
            str(item.get("value") or "").strip()
            for item in (temporal_dietary_constraints or [])
            if isinstance(item, dict) and str(item.get("value") or "").strip()
        ]
        if temporal:
            lines.append(f"阶段性饮食目标：{'、'.join(temporal[:20])}")
        return lines[:20]

    @staticmethod
    def _merge_preferences(*sources: dict[str, list[str]]) -> dict[str, list[str]]:
        merged = {
            key: []
            for key in ("likes", "dislikes", "allergens", "dietary_constraints", "available_ingredients")
        }
        for source in sources:
            for key, items in sanitize_preferences(source or {}).items():
                for item in items:
                    if key in {"likes", "dislikes"}:
                        opposite = "dislikes" if key == "likes" else "likes"
                        merged[opposite] = [
                            existing
                            for existing in merged[opposite]
                            if existing.lower() != item.lower()
                        ]
                    if item not in merged[key]:
                        merged[key].append(item)
        return sanitize_preferences(merged)

    def _compact_if_needed(self, memory: ConversationMemory) -> None:
        threshold_hit = (
            memory.turns_since_summary >= self.max_turns
            or memory.estimated_tokens >= self.max_tokens
        )
        if not threshold_hit or len(memory.recent_turns) <= self.keep_recent_messages:
            return
        older = memory.recent_turns[:-self.keep_recent_messages]
        memory.summary = summarize_incrementally(memory.summary, older, memory.preferences)
        memory.recent_turns = memory.recent_turns[-self.keep_recent_messages:]
        memory.turns_since_summary = 0
        memory.estimated_tokens = sum(max(1, len(turn.content) // 2) for turn in memory.recent_turns)

    async def _save(self, memory: ConversationMemory) -> None:
        memory.version += 1
        memory.updated_at = time.time()
        await self._save_conversation(
            memory,
            expires_at=memory.updated_at + self.ttl_seconds,
        )

    async def healthcheck(self) -> dict[str, bool]:
        result = {}
        for name, backend in (("conversation", self.store), ("profile", self.profile_store)):
            if name == "profile" and backend is self.store:
                result[name] = result.get("conversation", True)
                continue
            check = getattr(backend, "healthcheck", None)
            result[name] = bool(await check()) if check else True
            if result[name]:
                self._backend_failures.pop(name, None)
                self._backend_open_until.pop(name, None)
        return result

    async def close(self) -> None:
        await self.drain_pending(
            timeout_seconds=float(os.getenv("CONVERSATION_BACKGROUND_DRAIN_SECONDS", "2"))
        )
        closed = set()
        for backend in (self.store, self.profile_store):
            if id(backend) in closed:
                continue
            closed.add(id(backend))
            close = getattr(backend, "close", None)
            if close:
                await close()
        if self.profile_memory_index is not None:
            close = getattr(self.profile_memory_index, "close", None)
            if close:
                await close()


_service: ConversationService | None = None


def get_conversation_service() -> ConversationService:
    global _service
    if _service is not None:
        return _service
    mode = os.getenv("CONVERSATION_STORE", "redis").strip().lower()
    if mode == "memory":
        store = InMemoryConversationStore()
    elif mode == "redis":
        from app.conversation.redis_store import RedisConversationStore
        store = RedisConversationStore.from_env()
    else:
        raise ValueError(
            f"unsupported CONVERSATION_STORE={mode!r}; expected 'redis' "
            "or test-only 'memory'"
        )

    profile_default = "same" if mode == "memory" else "postgres"
    profile_mode = os.getenv("PROFILE_STORE", profile_default).strip().lower()
    if profile_mode in {"same", "conversation"}:
        if mode != "memory":
            raise ValueError(
                "PROFILE_STORE must be 'postgres' when CONVERSATION_STORE='redis'"
            )
        profile_store = store
    elif profile_mode == "postgres":
        from app.conversation.postgres_profile_store import PostgresChannelUserProfileStore
        profile_store = PostgresChannelUserProfileStore.from_env()
    else:
        raise ValueError(f"unsupported PROFILE_STORE={profile_mode!r}")

    profile_memory_index = None
    profile_fact_extractor = None
    if os.getenv("CONVERSATION_GENERAL_MEMORY_ENABLED", "false").lower() == "true":
        from app.conversation.milvus_memory_index import MilvusProfileMemoryIndex
        from app.conversation.profile_fact_extractor import ProfileFactExtractor

        profile_memory_index = MilvusProfileMemoryIndex.from_env(require_server=True)
        profile_fact_extractor = ProfileFactExtractor()

    _service = ConversationService(
        store,
        profile_store=profile_store,
        profile_memory_index=profile_memory_index,
        profile_fact_extractor=profile_fact_extractor,
        ttl_seconds=int(os.getenv("CONVERSATION_TTL_SECONDS", "86400")),
        max_turns=int(os.getenv("CONVERSATION_MAX_TURNS", "20")),
        keep_recent_turns=int(os.getenv("CONVERSATION_KEEP_RECENT_TURNS", "6")),
        max_tokens=int(os.getenv("CONVERSATION_MAX_TOKENS", "6000")),
        profile_ttl_seconds=int(os.getenv("CONVERSATION_PROFILE_TTL_SECONDS", "31536000")),
        transient_diet_ttl_seconds=int(
            os.getenv("CONVERSATION_TRANSIENT_DIET_TTL_SECONDS", "2592000")
        ),
        memory_recall_top_k=int(os.getenv("MEMORY_RECALL_TOP_K", "8")),
        memory_recall_min_score=float(
            os.getenv("MEMORY_RECALL_MIN_SCORE", "0.35")
        ),
        read_timeout_seconds=float(os.getenv("CONVERSATION_READ_TIMEOUT_SECONDS", "5")),
        write_timeout_seconds=float(os.getenv("CONVERSATION_WRITE_TIMEOUT_SECONDS", "5")),
        circuit_failure_threshold=int(os.getenv("CONVERSATION_CIRCUIT_FAILURE_THRESHOLD", "2")),
        circuit_cooldown_seconds=float(os.getenv("CONVERSATION_CIRCUIT_COOLDOWN_SECONDS", "30")),
    )
    return _service


def build_qq_thread_id(chat_type: str, chat_id: str, user_id: str) -> str:
    """群聊按用户隔离，避免不同成员共享候选、偏好或设备操作状态。"""
    return f"qq:{chat_type or 'unknown'}:{chat_id}:{user_id}"


def build_whatsapp_thread_id(chat_type: str, chat_id: str, user_id: str) -> str:
    """WhatsApp 私聊按号码、群聊按群组成员隔离，避免会话和偏好串线。"""
    return f"whatsapp:{chat_type or 'unknown'}:{chat_id or 'unknown'}:{user_id or 'unknown'}"


def build_weixin_thread_id(account_id: str, user_id: str) -> str:
    """微信按机器人账号和用户隔离；同一服务接多个微信账号时不会串会话。"""
    return f"weixin:dm:{account_id or 'default'}:{user_id or 'unknown'}"


def conversation_channel(thread_id: str) -> str:
    """从受控线程 ID 推断通道；旧调用没有前缀时维持 QQ 兼容默认值。"""
    prefix = str(thread_id or "").split(":", 1)[0].lower()
    return prefix if prefix in {"qq", "whatsapp", "weixin", "web"} else "qq"


def conversation_identity(thread_id: str) -> tuple[str, str | None]:
    """从受控 thread_id 中提取通道和账号；Web 永远没有账号身份。"""
    value = str(thread_id or "")
    channel = conversation_channel(value)
    if channel == "web":
        return "web", None
    identity = _parse_account_identity(value)
    return (identity[0], identity[2]) if identity else (channel, None)


def _parse_account_identity(
    thread_id: str,
) -> tuple[str, str | None, str] | None:
    """只从 canonical IM thread 提取账号身份；匿名/残缺 ID 永不映射画像。"""
    parts = str(thread_id or "").split(":", 3)
    if len(parts) != 4 or any(not part for part in parts):
        return None
    channel, scope, owner_id, user_id = parts
    channel = channel.lower()
    if channel not in {"qq", "whatsapp", "weixin"}:
        return None
    if channel == "weixin" and scope != "dm":
        return None
    account_id = owner_id if channel == "weixin" else None
    return channel, account_id, user_id


def supports_session_memory(thread_id: str) -> bool:
    """稳定 thread 可使用 Redis 短期会话；Web 仍保持匿名。"""
    prefix = str(thread_id or "").split(":", 1)[0].lower()
    return prefix in {"qq", "whatsapp", "weixin", "web"} and ":" in str(thread_id or "")


def supports_account_profile(thread_id: str) -> bool:
    """只有受控 IM thread 可映射到 PostgreSQL 账号画像。"""
    return _parse_account_identity(thread_id) is not None


def supports_persistent_memory(thread_id: str) -> bool:
    """兼容旧调用：此名称只代表账号级持久画像能力。"""
    return supports_account_profile(thread_id)


def supports_persistent_task_state(thread_id: str) -> bool:
    """稳定会话 ID 可持久化任务态；Web 不因此获得长期画像身份。"""
    value = str(thread_id or "")
    prefix = value.split(":", 1)[0].lower()
    return prefix in {"qq", "whatsapp", "weixin", "web"} and ":" in value
