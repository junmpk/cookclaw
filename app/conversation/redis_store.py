"""Redis 短期会话存储。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from app.conversation.models import ConversationMemory, ConversationTaskState
from app.conversation.store import ConversationStoreConflict
from app.conversation.task_state_store import (
    DeviceStartClaim,
    TaskStateRecord,
    TaskStateStoreConflict,
)
from app.core.env import getenv_resolved

_SAVE_IF_CURRENT = """
local current = redis.call('GET', KEYS[1])
local encoded = ARGV[1]
if current then
    local ok_current, current_value = pcall(cjson.decode, current)
    local ok_incoming, incoming_value = pcall(cjson.decode, ARGV[1])
    if not ok_current or not ok_incoming then
        return -2
    end
    local current_version = tonumber(current_value['version'] or 0)
    local incoming_version = tonumber(incoming_value['version'] or 0)
    local current_generation = tonumber(current_value['conversation_generation'] or 0)
    local incoming_generation = tonumber(incoming_value['conversation_generation'] or 0)
    if incoming_generation ~= current_generation then
        return 0
    end
    if incoming_version <= current_version then
        return 0
    end
    if tonumber(current_value['task_state_storage_version'] or 1) >= 2 then
        incoming_value['task_state_storage_version'] = 2
        incoming_value['task_state'] = current_value['task_state'] or {}
        encoded = cjson.encode(incoming_value)
    end
end
redis.call('SET', KEYS[1], encoded, 'EX', ARGV[2])
return 1
"""

_IMPORT_IF_NEWER = """
local current = redis.call('GET', KEYS[1])
local encoded = ARGV[1]
if current then
    local ok_current, current_value = pcall(cjson.decode, current)
    local ok_incoming, incoming_value = pcall(cjson.decode, ARGV[1])
    if not ok_current or not ok_incoming then
        return -2
    end
    local current_version = tonumber(current_value['version'] or 0)
    local incoming_version = tonumber(incoming_value['version'] or 0)
    local current_generation = tonumber(current_value['conversation_generation'] or 0)
    local incoming_generation = tonumber(incoming_value['conversation_generation'] or 0)
    if incoming_generation ~= current_generation then
        return 0
    end
    if incoming_version <= current_version then
        return 0
    end
    if tonumber(current_value['task_state_storage_version'] or 1) >= 2 then
        incoming_value['task_state_storage_version'] = 2
        incoming_value['task_state'] = current_value['task_state'] or {}
        encoded = cjson.encode(incoming_value)
    end
end
redis.call('SET', KEYS[1], encoded, 'EX', ARGV[2])
return 1
"""

_RESET_CONVERSATION = """
local current = redis.call('GET', KEYS[1])
local expected_version = tonumber(ARGV[2])
local expected_generation = tonumber(ARGV[3])
local current_version = 0
local current_generation = 0
if current then
    local current_ok, current_value = pcall(cjson.decode, current)
    if not current_ok or type(current_value) ~= 'table' then
        return {-2, 0, 0}
    end
    current_version = tonumber(current_value['version'] or 0)
    current_generation = tonumber(current_value['conversation_generation'] or 0)
end
if current_version ~= expected_version or current_generation ~= expected_generation then
    return {0, current_version, current_generation}
end
local incoming_ok, incoming = pcall(cjson.decode, ARGV[1])
if not incoming_ok or type(incoming) ~= 'table' then
    return {-2, current_version, current_generation}
end
incoming['version'] = current_version + 1
incoming['conversation_generation'] = current_generation + 1
redis.call('SET', KEYS[1], cjson.encode(incoming), 'EX', ARGV[4])
return {1, current_version + 1, current_generation + 1}
"""

_RESET_CONVERSATION_AND_TASK_STATE = """
local conversation_raw = redis.call('GET', KEYS[1])
local task_raw = redis.call('GET', KEYS[2])
local expected_version = tonumber(ARGV[3])
local expected_generation = tonumber(ARGV[4])
local expected_task_revision = tonumber(ARGV[5])
local expected_task_generation = tonumber(ARGV[6])

local current_version = 0
local current_generation = 0
if conversation_raw then
    local ok, value = pcall(cjson.decode, conversation_raw)
    if not ok or type(value) ~= 'table' then
        return {-2, 0, 0, 0, 0}
    end
    current_version = tonumber(value['version'] or 0)
    current_generation = tonumber(value['conversation_generation'] or 0)
end
if current_version ~= expected_version or current_generation ~= expected_generation then
    return {0, current_version, current_generation, 0, 0}
end

local current_task_revision = 0
local current_task_generation = 0
if task_raw then
    local ok, value = pcall(cjson.decode, task_raw)
    if not ok or type(value) ~= 'table' then
        return {-2, current_version, current_generation, 0, 0}
    end
    current_task_revision = tonumber(value['revision'] or 0)
    current_task_generation = tonumber(value['generation'] or 0)
end
if current_task_revision ~= expected_task_revision
    or current_task_generation ~= expected_task_generation then
    return {-1, current_version, current_generation, current_task_revision, current_task_generation}
end

local conversation_ok, incoming_conversation = pcall(cjson.decode, ARGV[1])
local state_ok, incoming_state = pcall(cjson.decode, ARGV[2])
if not conversation_ok or type(incoming_conversation) ~= 'table'
    or not state_ok or type(incoming_state) ~= 'table' then
    return {-2, current_version, current_generation, current_task_revision, current_task_generation}
end

local next_version = current_version + 1
local next_generation = current_generation + 1
local next_task_revision = current_task_revision + 1
local next_task_generation = current_task_generation + 1
incoming_conversation['version'] = next_version
incoming_conversation['conversation_generation'] = next_generation
incoming_conversation['task_state_storage_version'] = 2
incoming_conversation['task_state'] = incoming_state
local task_record = {
    ['thread_id'] = ARGV[9],
    ['state'] = incoming_state,
    ['revision'] = next_task_revision,
    ['generation'] = next_task_generation,
    ['updated_at'] = tonumber(ARGV[8])
}
redis.call('SET', KEYS[1], cjson.encode(incoming_conversation), 'EX', ARGV[7])
redis.call('SET', KEYS[2], cjson.encode(task_record), 'EX', ARGV[7])
return {1, next_version, next_generation, next_task_revision, next_task_generation}
"""

_CONSUME_PENDING_DEVICE_START = """
local current = redis.call('GET', KEYS[1])
if not current then
    return {0, ''}
end
local ok_current, current_value = pcall(cjson.decode, current)
if not ok_current then
    return {-2, ''}
end
local task = current_value['task_state']
if type(task) ~= 'table' then
    return {0, ''}
end
local execution = task['device_execution']
if type(execution) == 'table' and next(execution) ~= nil then
    local execution_status = tostring(execution['status'] or '')
    if execution_status == 'dispatching'
        or execution_status == 'submitted_unverified'
        or execution_status == 'outcome_unknown' then
        return {2, ''}
    end
end
local pending = task['pending_device_start']
if type(pending) ~= 'table' or next(pending) == nil then
    return {0, ''}
end
if tostring(pending['action_id'] or '') ~= ARGV[1] then
    return {0, ''}
end

local now = tonumber(ARGV[2])
local next_execution = {}
for key, value in pairs(pending) do
    next_execution[key] = value
end
next_execution['status'] = 'dispatching'
next_execution['confirmation_created_at'] = tonumber(pending['ts'] or 0)
next_execution['claimed_at'] = now
next_execution['updated_at'] = now
next_execution['ts'] = now
next_execution['result_code'] = ''
next_execution['command_sent'] = cjson.null
next_execution['started'] = cjson.null
task['device_execution'] = next_execution
task['pending_device_start'] = cjson.null
local pending_action = task['pending_action']
if type(pending_action) == 'table' then
    local kind = tostring(pending_action['kind'] or '')
    if kind == 'choose_device' or kind == 'confirm_device_start' then
        task['pending_action'] = cjson.null
    end
end
if type(task['active_cooking']) == 'table' and next(task['active_cooking']) ~= nil then
    task['current_task'] = 'device_cooking'
elseif type(task['device_execution']) == 'table'
    and tostring(task['device_execution']['status'] or '') == 'dispatching' then
    task['current_task'] = 'device_execution'
elseif type(task['candidate_recipes']) == 'table' and #task['candidate_recipes'] > 0 then
    task['current_task'] = 'recipe_selection'
elseif type(task['active_search_request']) == 'table' and next(task['active_search_request']) ~= nil then
    task['current_task'] = 'recipe_search'
else
    task['current_task'] = cjson.null
end
task['selected_device_id'] = tostring(next_execution['device_id'] or '') ~= ''
    and tostring(next_execution['device_id']) or cjson.null
local language = tostring(next_execution['lang'] or '')
task['language'] = (language == 'zh' or language == 'en') and language or cjson.null
task['updated_at'] = now
current_value['version'] = tonumber(current_value['version'] or 0) + 1
current_value['updated_at'] = now

local encoded = cjson.encode(current_value)
redis.call('SET', KEYS[1], encoded, 'KEEPTTL')
return {1, cjson.encode(pending)}
"""

_COMMIT_TASK_STATE_RECORD = """
local current = redis.call('GET', KEYS[1])
local expected_revision = tonumber(ARGV[2])
local expected_generation = tonumber(ARGV[5])
local incoming_ok, incoming = pcall(cjson.decode, ARGV[1])
if not incoming_ok or type(incoming) ~= 'table' then
    return {-2, 0, 0}
end
local current_generation = 0
if current then
    local current_ok, current_value = pcall(cjson.decode, current)
    if not current_ok or type(current_value) ~= 'table' then
        return {-2, 0, 0}
    end
    local current_revision = tonumber(current_value['revision'] or 0)
    current_generation = tonumber(current_value['generation'] or 0)
    if current_revision ~= expected_revision then
        return {0, current_revision, current_generation}
    end
    if current_generation ~= expected_generation then
        return {0, current_revision, current_generation}
    end
elseif expected_revision ~= 0 then
    return {0, 0, 0}
elseif expected_generation ~= 0 then
    return {0, 0, 0}
end
local target_generation = tonumber(ARGV[6])
if target_generation < 0 then
    target_generation = current_generation
end
if target_generation < current_generation or target_generation > current_generation + 1 then
    return {-3, expected_revision, current_generation}
end
local next_revision = expected_revision + 1
incoming['revision'] = next_revision
incoming['generation'] = target_generation
incoming['updated_at'] = tonumber(ARGV[4])
redis.call('SET', KEYS[1], cjson.encode(incoming), 'EX', ARGV[3])
local conversation = redis.call('GET', KEYS[2])
if conversation then
    local conversation_ok, conversation_value = pcall(cjson.decode, conversation)
    if conversation_ok and type(conversation_value) == 'table' then
        conversation_value['task_state_storage_version'] = 2
        conversation_value['task_state'] = incoming['state'] or {}
        local conversation_encoded = cjson.encode(conversation_value)
        redis.call('SET', KEYS[2], conversation_encoded, 'KEEPTTL')
    end
end
return {1, next_revision, target_generation}
"""

_CLAIM_TASK_STATE_PENDING_DEVICE = """
local current = redis.call('GET', KEYS[1])
if not current then
    return {0, 0, '', ''}
end
local current_ok, record = pcall(cjson.decode, current)
if not current_ok or type(record) ~= 'table' then
    return {-2, 0, '', ''}
end
local revision = tonumber(record['revision'] or 0)
local generation = tonumber(record['generation'] or 0)
local state = record['state']
if type(state) ~= 'table' then
    return {-2, revision, '', '', generation}
end
local execution = state['device_execution']
if type(execution) == 'table' and next(execution) ~= nil then
    local execution_status = tostring(execution['status'] or '')
    if execution_status == 'dispatching'
        or execution_status == 'submitted_unverified'
        or execution_status == 'outcome_unknown' then
        return {4, revision, '', cjson.encode(state), generation}
    end
end
local pending = state['pending_device_start']
if type(pending) ~= 'table' or next(pending) == nil then
    return {0, revision, '', cjson.encode(state), generation}
end
if tostring(pending['action_id'] or '') ~= ARGV[1] then
    return {2, revision, '', cjson.encode(state), generation}
end

local now = tonumber(ARGV[2])
local max_age = tonumber(ARGV[3])
local created_at = tonumber(pending['ts'] or 0)
local status = 1
local payload = cjson.encode(pending)
if created_at <= 0 or now - created_at > max_age then
    status = 3
    payload = ''
else
    local next_execution = {}
    for key, value in pairs(pending) do
        next_execution[key] = value
    end
    next_execution['status'] = 'dispatching'
    next_execution['confirmation_created_at'] = created_at
    next_execution['claimed_at'] = now
    next_execution['updated_at'] = now
    next_execution['ts'] = now
    next_execution['result_code'] = ''
    next_execution['command_sent'] = cjson.null
    next_execution['started'] = cjson.null
    state['device_execution'] = next_execution
    execution = next_execution
end

state['pending_device_start'] = cjson.null
local pending_action = state['pending_action']
if type(pending_action) == 'table' then
    local kind = tostring(pending_action['kind'] or '')
    if kind == 'choose_device' or kind == 'confirm_device_start' then
        state['pending_action'] = cjson.null
    end
end
if type(state['active_cooking']) == 'table' and next(state['active_cooking']) ~= nil then
    state['current_task'] = 'device_cooking'
elseif type(state['pending_device_start']) == 'table' then
    state['current_task'] = 'device_start'
elseif type(execution) == 'table'
    and (tostring(execution['status'] or '') == 'dispatching'
        or tostring(execution['status'] or '') == 'submitted_unverified'
        or tostring(execution['status'] or '') == 'outcome_unknown') then
    state['current_task'] = 'device_execution'
elseif type(state['pending_search_clarification']) == 'table' and next(state['pending_search_clarification']) ~= nil then
    state['current_task'] = 'recipe_search'
elseif type(state['menu_task']) == 'table' and next(state['menu_task']) ~= nil then
    state['current_task'] = 'menu_plan'
elseif type(state['candidate_recipes']) == 'table' and #state['candidate_recipes'] > 0 then
    state['current_task'] = 'recipe_selection'
elseif type(state['active_search_request']) == 'table' and next(state['active_search_request']) ~= nil then
    state['current_task'] = 'recipe_search'
elseif type(state['focus']) == 'table' and next(state['focus']) ~= nil then
    state['current_task'] = 'recipe_discussion'
else
    state['current_task'] = cjson.null
end
local selected_device_id = ''
if type(state['pending_device_start']) == 'table' then
    selected_device_id = tostring(state['pending_device_start']['device_id'] or '')
end
if selected_device_id == '' and type(state['active_cooking']) == 'table' then
    selected_device_id = tostring(state['active_cooking']['device_id'] or '')
end
if selected_device_id == '' and type(execution) == 'table' then
    selected_device_id = tostring(execution['device_id'] or '')
end
state['selected_device_id'] = (
    selected_device_id ~= '' and selected_device_id or cjson.null
)
local language = ''
if type(state['pending_device_start']) == 'table' then
    language = tostring(state['pending_device_start']['lang'] or '')
end
if language ~= 'zh' and language ~= 'en' and type(execution) == 'table' then
    language = tostring(execution['lang'] or '')
end
if language ~= 'zh' and language ~= 'en' and type(state['selected_recipe']) == 'table' then
    language = tostring(state['selected_recipe']['lang'] or '')
end
if language ~= 'zh' and language ~= 'en' then
    language = tostring(state['candidate_language'] or '')
end
if language ~= 'zh' and language ~= 'en' and type(state['focus']) == 'table' then
    language = tostring(state['focus']['lang'] or '')
end
if language ~= 'zh' and language ~= 'en' and type(state['active_cooking']) == 'table' then
    language = tostring(state['active_cooking']['lang'] or '')
end
if language ~= 'zh' and language ~= 'en' then
    language = tostring(state['language'] or '')
end
state['language'] = (
    (language == 'zh' or language == 'en') and language or cjson.null
)
state['updated_at'] = now
record['revision'] = revision + 1
record['updated_at'] = now

local encoded = cjson.encode(record)
redis.call('SET', KEYS[1], encoded, 'KEEPTTL')
return {status, revision + 1, payload, cjson.encode(state), generation}
"""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = getenv_resolved(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class RedisConversationStore:
    """以 JSON 字符串保存 ConversationMemory，并由 Redis TTL 自动清理。"""

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = "cookclaw:demo",
        enforce_version: bool = True,
    ) -> None:
        self.client = client
        self.key_prefix = key_prefix.strip().strip(":") or "cookclaw:demo"
        self.enforce_version = enforce_version

    @classmethod
    def from_env(cls) -> "RedisConversationStore":
        profile = (
            getenv_resolved("DATA_SERVICE_PROFILE", "local") or "local"
        ).strip().lower()
        prefix = "REDIS_SERVER_" if profile == "server" else "REDIS_"
        host = (getenv_resolved(f"{prefix}HOST", "") or "").strip()
        port = int(getenv_resolved(f"{prefix}PORT", "6379") or "6379")
        password = getenv_resolved(f"{prefix}PASSWORD", "") or ""
        database = int(getenv_resolved(f"{prefix}DB", "0") or "0")
        ssl_enabled = _env_bool(f"{prefix}SSL", False)
        if not host:
            raise ValueError(f"{prefix}HOST is required for Redis conversation storage")
        retry_attempts = max(0, min(2, int(os.getenv("REDIS_RETRY_ATTEMPTS", "0"))))
        client = Redis(
            host=host,
            port=port,
            password=password or None,
            db=database,
            ssl=ssl_enabled,
            decode_responses=True,
            socket_connect_timeout=float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS", "5")),
            socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "5")),
            retry=Retry(NoBackoff(), retries=retry_attempts),
            health_check_interval=30,
        )
        enforce_version = _env_bool("REDIS_CONVERSATION_CAS", True)
        if not enforce_version:
            raise ValueError(
                "REDIS_CONVERSATION_CAS=false is unsafe and no longer supported"
            )
        return cls(
            client,
            key_prefix=os.getenv("REDIS_KEY_PREFIX", "cookclaw:demo"),
            enforce_version=True,
        )

    def key_for(self, thread_id: str) -> str:
        digest = hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:conversation:{digest}"

    def task_state_key_for(self, thread_id: str) -> str:
        digest = hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:task-state:{digest}"

    @staticmethod
    def _serialize(memory: ConversationMemory) -> str:
        return json.dumps(memory.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _deserialize(payload: str) -> ConversationMemory:
        value: Any = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Redis conversation payload must be a JSON object")
        return ConversationMemory.from_dict(value)

    async def load(self, thread_id: str) -> ConversationMemory | None:
        payload = await self.client.get(self.key_for(thread_id))
        if payload is None:
            return None
        memory = self._deserialize(payload)
        if memory.thread_id != thread_id:
            raise ValueError("Redis conversation key/payload mismatch")
        return memory

    async def save(self, memory: ConversationMemory, expires_at: float) -> None:
        ttl = max(1, math.ceil(float(expires_at) - time.time()))
        key = self.key_for(memory.thread_id)
        payload = self._serialize(memory)
        result = int(await self.client.eval(_SAVE_IF_CURRENT, 1, key, payload, ttl))
        if result == 0:
            raise ConversationStoreConflict(
                f"conversation version conflict for {key[-12:]}"
            )
        if result != 1:
            raise ValueError("Redis conversation payload could not be version checked")

    async def reset(
        self,
        memory: ConversationMemory,
        *,
        expected_version: int,
        expected_generation: int,
        expires_at: float,
    ) -> None:
        """原子写入新一代空会话墓碑，拒绝旧 worker 跨代保存。"""
        ttl = max(1, math.ceil(float(expires_at) - time.time()))
        result = await self.client.eval(
            _RESET_CONVERSATION,
            1,
            self.key_for(memory.thread_id),
            self._serialize(memory),
            max(0, int(expected_version)),
            max(0, int(expected_generation)),
            ttl,
        )
        status = int(result[0]) if isinstance(result, (list, tuple)) else -2
        if status == 0:
            actual_version = int(result[1]) if len(result) > 1 else 0
            actual_generation = int(result[2]) if len(result) > 2 else 0
            raise ConversationStoreConflict(
                "conversation reset baseline changed: "
                f"version={actual_version} generation={actual_generation}"
            )
        if status != 1:
            raise ValueError("Redis conversation payload could not be reset")

    async def reset_with_task_state(
        self,
        memory: ConversationMemory,
        task_state: ConversationTaskState,
        *,
        expected_version: int,
        expected_generation: int,
        expected_task_revision: int,
        expected_task_generation: int,
        expires_at: float,
    ) -> TaskStateRecord:
        """用单个 Lua 同时推进 conversation/task generation，不允许半重置。"""
        ttl = max(1, math.ceil(float(expires_at) - time.time()))
        updated_at = time.time()
        result = await self.client.eval(
            _RESET_CONVERSATION_AND_TASK_STATE,
            2,
            self.key_for(memory.thread_id),
            self.task_state_key_for(memory.thread_id),
            self._serialize(memory),
            json.dumps(
                task_state.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            max(0, int(expected_version)),
            max(0, int(expected_generation)),
            max(0, int(expected_task_revision)),
            max(0, int(expected_task_generation)),
            ttl,
            updated_at,
            memory.thread_id,
        )
        status = int(result[0]) if isinstance(result, (list, tuple)) else -2
        if status == 0:
            actual_version = int(result[1]) if len(result) > 1 else 0
            actual_generation = int(result[2]) if len(result) > 2 else 0
            raise ConversationStoreConflict(
                "conversation reset baseline changed: "
                f"version={actual_version} generation={actual_generation}"
            )
        if status == -1:
            actual_revision = int(result[3]) if len(result) > 3 else 0
            actual_generation = int(result[4]) if len(result) > 4 else 0
            raise TaskStateStoreConflict(
                "task state reset baseline changed: "
                f"revision={actual_revision} generation={actual_generation}"
            )
        if status != 1:
            raise ValueError("Redis short-term state could not be reset atomically")
        return TaskStateRecord(
            thread_id=memory.thread_id,
            state=ConversationTaskState.from_dict(task_state.to_dict()),
            revision=int(result[3]),
            generation=int(result[4]),
            updated_at=updated_at,
        )

    async def import_if_newer(
        self,
        memory: ConversationMemory,
        expires_at: float,
    ) -> bool:
        """迁移专用：仅当目标不存在或源版本更新时写入，支持安全重跑。"""
        ttl = max(1, math.ceil(float(expires_at) - time.time()))
        key = self.key_for(memory.thread_id)
        result = int(
            await self.client.eval(
                _IMPORT_IF_NEWER,
                1,
                key,
                self._serialize(memory),
                ttl,
            )
        )
        if result == -2:
            raise ValueError("Redis conversation payload could not be version checked")
        return result == 1

    async def consume_pending_device_start(
        self,
        thread_id: str,
        expected_action_id: str,
    ) -> dict | None:
        """原子读取并删除待确认启动动作，供多 worker 竞争同一条确认。"""
        result = await self.client.eval(
            _CONSUME_PENDING_DEVICE_START,
            1,
            self.key_for(thread_id),
            str(expected_action_id),
            time.time(),
        )
        status = int(result[0]) if isinstance(result, (list, tuple)) and result else -2
        if status in {0, 2}:
            return None
        if status != 1:
            raise ValueError("Redis pending device action could not be consumed")
        payload = result[1] if len(result) > 1 else ""
        value: Any = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Redis pending device action must be a JSON object")
        return value

    async def load_task_state_record(
        self,
        thread_id: str,
    ) -> TaskStateRecord | None:
        payload = await self.client.get(self.task_state_key_for(thread_id))
        if payload is None:
            return None
        value: Any = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Redis task state payload must be a JSON object")
        if str(value.get("thread_id") or "") != str(thread_id):
            raise ValueError("Redis task state key/payload mismatch")
        return TaskStateRecord(
            thread_id=str(thread_id),
            state=ConversationTaskState.from_dict(value.get("state")),
            revision=max(0, int(value.get("revision") or 0)),
            generation=max(0, int(value.get("generation") or 0)),
            updated_at=max(0.0, float(value.get("updated_at") or 0)),
        )

    async def commit_task_state_record(
        self,
        record: TaskStateRecord,
        *,
        expected_revision: int,
        expected_generation: int,
        new_generation: int | None,
        expires_at: float,
    ) -> TaskStateRecord:
        ttl = max(1, math.ceil(float(expires_at) - time.time()))
        updated_at = max(time.time(), float(record.updated_at or 0))
        payload = json.dumps(
            {
                "thread_id": record.thread_id,
                "state": record.state.to_dict(),
                "revision": max(0, int(expected_revision)),
                "generation": (
                    max(0, int(expected_generation))
                    if new_generation is None
                    else max(0, int(new_generation))
                ),
                "updated_at": updated_at,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        # 任务状态承载设备确认和 reset 代际屏障，无论普通会话是否
        # 允许关闭 version CAS，此处都必须走 Redis Lua CAS。
        result = await self.client.eval(
            _COMMIT_TASK_STATE_RECORD,
            2,
            self.task_state_key_for(record.thread_id),
            self.key_for(record.thread_id),
            payload,
            max(0, int(expected_revision)),
            ttl,
            updated_at,
            max(0, int(expected_generation)),
            (
                -1
                if new_generation is None
                else max(0, int(new_generation))
            ),
        )
        status = int(result[0]) if isinstance(result, (list, tuple)) and result else -2
        actual_revision = (
            int(result[1])
            if isinstance(result, (list, tuple)) and len(result) > 1
            else 0
        )
        actual_generation = (
            int(result[2])
            if isinstance(result, (list, tuple)) and len(result) > 2
            else 0
        )
        if status == 0:
            raise TaskStateStoreConflict(
                "task state revision conflict: "
                f"expected={expected_revision} actual={actual_revision}; "
                "generation "
                f"expected={expected_generation} actual={actual_generation}"
            )
        if status != 1:
            raise ValueError("Redis task state payload could not be committed")
        return TaskStateRecord(
            thread_id=record.thread_id,
            state=ConversationTaskState.from_dict(record.state.to_dict()),
            revision=actual_revision,
            generation=actual_generation,
            updated_at=updated_at,
        )

    async def claim_pending_device_start_record(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
        max_age_seconds: int,
        now: float,
    ) -> DeviceStartClaim:
        result = await self.client.eval(
            _CLAIM_TASK_STATE_PENDING_DEVICE,
            1,
            self.task_state_key_for(thread_id),
            str(expected_action_id),
            float(now),
            max(1, int(max_age_seconds)),
        )
        status_code = (
            int(result[0])
            if isinstance(result, (list, tuple)) and result
            else -2
        )
        revision = (
            int(result[1])
            if isinstance(result, (list, tuple)) and len(result) > 1
            else 0
        )
        if status_code < 0:
            raise ValueError("Redis task state pending action could not be consumed")
        status = {
            0: "missing",
            1: "claimed",
            2: "mismatch",
            3: "expired",
            4: "in_progress",
        }.get(status_code)
        if status is None:
            raise ValueError("Redis task state claim returned an unknown status")
        raw_payload = result[2] if len(result) > 2 else ""
        payload: dict | None = None
        if status == "claimed":
            value: Any = json.loads(raw_payload)
            if not isinstance(value, dict):
                raise ValueError("Redis pending device action must be a JSON object")
            payload = value
        raw_state = result[3] if len(result) > 3 else ""
        state: ConversationTaskState | None = None
        if raw_state:
            state_value: Any = json.loads(raw_state)
            if not isinstance(state_value, dict):
                raise ValueError("Redis task state claim snapshot must be a JSON object")
            state = ConversationTaskState.from_dict(state_value)
        generation = (
            int(result[4])
            if isinstance(result, (list, tuple)) and len(result) > 4
            else 0
        )
        return DeviceStartClaim(status, payload, revision, state, generation)

    async def delete_task_state_record(self, thread_id: str) -> None:
        await self.client.delete(self.task_state_key_for(thread_id))

    async def delete(self, thread_id: str) -> None:
        await self.client.delete(self.key_for(thread_id))

    async def cleanup_expired(self, now: float | None = None) -> int:
        # Redis 原生 TTL 自动清理；保留协议方法以兼容现有 Store 接口。
        return 0

    async def healthcheck(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        await self.client.aclose()
