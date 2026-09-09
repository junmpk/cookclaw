"""
会话态 —— 支撑「搜 → 选 → 执行」状态机的同步访问层（进程缓存 + TTL）。

按 thread_id（通道会话，如 QQ chat_id）隔离这些短期状态：
  - 上次检索的候选列表（供"回复编号/做第X个/做<菜名>"选择）；
  - 推荐前正在等待用户补充的结构化条件；
  - 待确认开火的 pending（选中某菜后、用户确认前）。

cookId 取候选的 metadata.recipe_id（脊梁 recipe_id ≡ 设备 cookId）。
持久化通道会在每轮开始/结束时从 ConversationMemory.task_state 恢复和保存；
这些字典只作为单轮同步业务代码的兼容缓存，不再是生产事实源。
⚠️ 执行目标仍是写死的 provider_device_id（单设备 demo）；多用户「账号→SN」绑定后续做。
"""
from __future__ import annotations

import re
import time
from typing import Optional

from app.conversation.models import ConversationTaskState
from app.domain.device_execution import (
    new_device_action_identity,
    normalize_device_msg_id,
)

_CANDIDATE_TTL = 600   # 候选有效期（秒）
_PENDING_TTL = 300     # 待确认开火有效期（秒）
_PENDING_ACTION_TTL = 900  # 普通下一步动作有效期（秒）
_FOCUS_TTL = 900       # 最近讨论主题有效期（秒）
_CLARIFICATION_TTL = 600  # 推荐前待补条件有效期（秒）
_ACTIVE_TTL = 21_600   # 运行中任务不能沿用 10 分钟候选 TTL
_EXECUTION_TTL = 86_400  # 未决设备结果至少保留到短期会话物理 TTL

_last: dict[str, dict] = {}      # thread_id -> {"items": [{"cookId","name"}], "lang": "zh|en", "ts": float}
_pending: dict[str, dict] = {}   # thread_id -> {"cookId","name","ts": float}
_active: dict[str, dict] = {}    # thread_id -> {"cookId","name","device_id","lang","ts": float}
_device_execution: dict[str, dict] = {}  # thread_id -> 持久设备执行状态机工作副本
_focus: dict[str, dict] = {}     # thread_id -> {"topic","lang","source","ts": float}
_search_clarification: dict[str, dict] = {}  # thread_id -> request/dimension/asked_dimensions/round_count/lang/ts
_pending_action: dict[str, dict] = {}  # thread_id -> {"kind","payload","lang","ts"}
_menu_task: dict[str, dict] = {}  # thread_id -> {"request","search_result","lang","ts"}
_selected: dict[str, dict] = {}  # thread_id -> {"cookId","name",...,"ts"}
_excluded: dict[str, dict] = {}  # thread_id -> {"ids": [...], "ts"}
_active_search: dict[str, dict] = {}  # thread_id -> {"request","lang","ts"}

_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_EN = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_DEVICE_NAME_ALIASES_EN = {
    "办公室设备": ("Office Device", "Office"),
    "展厅设备": ("Showroom Device", "Showroom"),
    "默认设备": ("Default Device", "Default"),
}


def remember_candidates(thread_id: str, search_result: dict, lang: str = None) -> None:
    """检索成功后，按展示顺序记住候选（cookId = metadata.recipe_id）。"""
    items = []
    for r in (search_result or {}).get("results", []):
        md = r.get("metadata", {})
        cid = str(md.get("recipe_id") or r.get("id") or "").strip()
        if cid:
            item = {"cookId": cid, "name": md.get("name", "")}
            if isinstance(r.get("score"), (int, float)):
                item["score"] = float(r["score"])
            for key in (
                "tags", "ingredients", "seasonings", "image_url", "description", "facets",
                "recipe_detail",
            ):
                value = md.get(key)
                if value:
                    item[key] = value
            if r.get("menu_role"):
                item["menu_role"] = r["menu_role"]
            items.append(item)
    request = (search_result or {}).get("_search_request") or {}
    if isinstance(request, dict) and request:
        set_active_search_request(thread_id, request, lang=lang)
    if items:
        # 写入新候选批次即开始新的选择上下文；排除和选择只属于上一批，
        # 不能按序号带到新批次。
        _selected.pop(thread_id, None)
        _excluded.pop(thread_id, None)
        _last[thread_id] = {
            "items": items,
            "lang": lang if lang in ("zh", "en") else None,
            "decision_id": str((search_result or {}).get("_decision_id") or ""),
            "search_request": dict((search_result or {}).get("_search_request") or {}),
            "ts": time.time(),
        }
        if len(items) == 1:
            remember_recipe_focus(thread_id, items[0], lang=lang, source="single_search_result")
        elif (_focus.get(thread_id) or {}).get("verified_recipe"):
            _focus.pop(thread_id, None)


def remember_recipe_focus(thread_id: str, recipe: dict, lang: str = None, source: str = "selection") -> None:
    """只把真实检索候选写成单菜焦点，禁止用查询词冒充已验证菜谱。"""
    recipe_id = str((recipe or {}).get("cookId") or (recipe or {}).get("id") or "").strip()
    name = str((recipe or {}).get("name") or "").strip()
    if not recipe_id or not name:
        return
    _focus[thread_id] = {
        "topic": name,
        "recipe_id": recipe_id,
        "name": name,
        "verified_recipe": True,
        "focus_kind": "recipe",
        "lang": lang if lang in ("zh", "en") else None,
        "source": source,
        "ts": time.time(),
    }


def recall_candidates(thread_id: str, max_age: int = _CANDIDATE_TTL) -> Optional[list]:
    rec = _last.get(thread_id)
    if not rec or time.time() - rec["ts"] > max_age:
        return None
    return rec["items"]


def recall_candidate_lang(thread_id: str, max_age: int = _CANDIDATE_TTL) -> Optional[str]:
    rec = _last.get(thread_id)
    if not rec or time.time() - rec["ts"] > max_age:
        return None
    return rec.get("lang")


def recall_candidate_context(thread_id: str, max_age: int = _CANDIDATE_TTL) -> Optional[dict]:
    """返回最近一次检索上下文，供"这几道/刚才推荐"这类追问使用。"""
    rec = _last.get(thread_id)
    if not rec or time.time() - rec["ts"] > max_age:
        return None
    return {
        "items": rec["items"],
        "lang": rec.get("lang"),
        "decision_id": rec.get("decision_id") or "",
        "search_request": dict(
            (get_active_search_request(thread_id) or {}).get("request")
            or rec.get("search_request")
            or {}
        ),
        "ts": rec["ts"],
    }


def set_active_search_request(
    thread_id: str,
    request: dict,
    *,
    lang: str | None = None,
    ts: float | None = None,
) -> None:
    if not thread_id or not isinstance(request, dict) or not request:
        return
    _active_search[thread_id] = {
        "request": dict(request),
        "lang": lang if lang in ("zh", "en") else None,
        "ts": float(ts or time.time()),
    }
    if thread_id in _last:
        _last[thread_id]["search_request"] = dict(request)


def get_active_search_request(
    thread_id: str,
    max_age: int = _ACTIVE_TTL,
) -> Optional[dict]:
    record = _active_search.get(thread_id)
    if not record:
        return None
    if time.time() - float(record.get("ts") or 0) > max_age:
        _active_search.pop(thread_id, None)
        return None
    return {
        "request": dict(record.get("request") or {}),
        "lang": record.get("lang"),
        "ts": float(record.get("ts") or 0),
    }


def clear_candidate_context(
    thread_id: str,
    *,
    keep_search_request: bool = True,
) -> None:
    """使旧候选失效；搜索条件可保留用于失败重试。"""
    _last.pop(thread_id, None)
    _selected.pop(thread_id, None)
    _excluded.pop(thread_id, None)
    _focus.pop(thread_id, None)
    if not keep_search_request:
        _active_search.pop(thread_id, None)


def set_selected_recipe(
    thread_id: str,
    recipe: dict,
    *,
    lang: str | None = None,
    source: str = "explicit_selection",
) -> None:
    """记录用户已经明确选中的真实候选；查看详情本身不等于选择。"""
    recipe_id = str(
        (recipe or {}).get("cookId") or (recipe or {}).get("id") or ""
    ).strip()
    name = str((recipe or {}).get("name") or "").strip()
    if not thread_id or not recipe_id or not name:
        return
    item = dict(recipe)
    item["cookId"] = recipe_id
    item["name"] = name
    item["lang"] = lang if lang in ("zh", "en") else item.get("lang")
    item["source"] = str(source or "explicit_selection")
    item["ts"] = time.time()
    _selected[thread_id] = item
    set_candidate_excluded(thread_id, recipe_id, excluded=False)
    remember_recipe_focus(
        thread_id,
        item,
        lang=item.get("lang"),
        source=item["source"],
    )


def get_selected_recipe(
    thread_id: str,
    max_age: int = _FOCUS_TTL,
) -> Optional[dict]:
    selected = _selected.get(thread_id)
    if not selected:
        return None
    if time.time() - float(selected.get("ts") or 0) > max_age:
        _selected.pop(thread_id, None)
        return None
    return dict(selected)


def clear_selected_recipe(thread_id: str) -> None:
    _selected.pop(thread_id, None)


def set_candidate_excluded(
    thread_id: str,
    recipe_id: str,
    *,
    excluded: bool = True,
) -> None:
    """按 recipe_id 标记排除，不删除候选，保证后续序号不会漂移。"""
    value = str(recipe_id or "").strip()
    if not thread_id or not value:
        return
    record = _excluded.get(thread_id) or {"ids": [], "ts": time.time()}
    ids = [str(item) for item in record.get("ids") or [] if str(item)]
    if excluded and value not in ids:
        ids.append(value)
    elif not excluded:
        ids = [item for item in ids if item != value]
    if ids:
        _excluded[thread_id] = {"ids": ids[-20:], "ts": time.time()}
    else:
        _excluded.pop(thread_id, None)
    selected = get_selected_recipe(thread_id)
    if excluded and selected and str(selected.get("cookId") or "") == value:
        clear_selected_recipe(thread_id)


def get_excluded_recipe_ids(
    thread_id: str,
    max_age: int = _FOCUS_TTL,
) -> list[str]:
    record = _excluded.get(thread_id)
    if not record:
        return []
    if time.time() - float(record.get("ts") or 0) > max_age:
        _excluded.pop(thread_id, None)
        return []
    return list(record.get("ids") or [])


def clear_candidate_decisions(thread_id: str) -> None:
    """新搜索或明确换题时清除上一批的选择/排除，不修改候选本身。"""
    _selected.pop(thread_id, None)
    _excluded.pop(thread_id, None)


def remember_focus(
    thread_id: str,
    topic: str,
    lang: str = None,
    source: str = "chat",
    focus_kind: str = "topic",
) -> None:
    """记录最近讨论对象，供"这个菜/它/刚才那个"这类单数指代使用。"""
    topic = (topic or "").strip()
    if not topic:
        return
    _focus[thread_id] = {
        "topic": topic,
        "focus_kind": focus_kind if focus_kind in {"topic", "query"} else "topic",
        "lang": lang if lang in ("zh", "en") else None,
        "source": source,
        "ts": time.time(),
    }


def recall_focus(thread_id: str, max_age: int = _FOCUS_TTL) -> Optional[dict]:
    rec = _focus.get(thread_id)
    if not rec or time.time() - rec["ts"] > max_age:
        return None
    return rec


def set_search_clarification(
    thread_id: str,
    request: dict,
    *,
    dimension: str,
    lang: str = None,
    asked_dimensions: list[str] | None = None,
    round_count: int = 1,
    ts: float | None = None,
) -> None:
    """记住本轮没搜、正在等用户补一句的结构化推荐需求。"""
    if not thread_id or not isinstance(request, dict):
        return
    dimensions = [
        str(item).strip()
        for item in (asked_dimensions or [])
        if str(item).strip()
    ]
    _search_clarification[thread_id] = {
        "request": dict(request),
        "dimension": str(dimension or ""),
        "asked_dimensions": dimensions,
        "round_count": max(1, int(round_count or 1)),
        "lang": lang if lang in ("zh", "en") else None,
        "ts": float(ts or time.time()),
    }


def get_search_clarification(
    thread_id: str,
    max_age: int = _CLARIFICATION_TTL,
) -> Optional[dict]:
    rec = _search_clarification.get(thread_id)
    if not rec:
        return None
    if time.time() - rec["ts"] > max_age:
        _search_clarification.pop(thread_id, None)
        return None
    return dict(rec)


def clear_search_clarification(thread_id: str) -> None:
    _search_clarification.pop(thread_id, None)


def _parse_ordinal(t: str) -> Optional[int]:
    """从文本解析出 1-based 序号；长数字（cookId）不当序号。"""
    t = (t or "").strip()
    tl = t.lower()
    if t in _CN:                       # 裸中文数字 "一/二/三"
        return _CN[t]
    if tl in _EN:                      # 裸英文序号 "first/second"
        return _EN[tl]
    if t.isdigit() and len(t) <= 2:    # 裸短数字 "1/2/3"（长数字是 cookId）
        return int(t)
    m = re.search(
        r"(?:菜谱|食谱|选项|方案)\s*[#：:]?\s*"
        r"(10|[1-9]|[一二三四五六七八九十])(?:\s*[个道号])?",
        t,
    ) or re.search(
        r"\b(?:recipe|option)\s*[#:]?\s*(10|[1-9])\b",
        tl,
    ) or re.search(r"第\s*(10|[1-9]|[一二三四五六七八九十])\s*[个道号]?", t) \
        or re.search(r"做\s*第?\s*(10|[1-9]|[一二三四五六七八九十])\s*[个道号]?", t)
    if m:
        g = m.group(1)
        return int(g) if g.isdigit() else _CN.get(g)
    m = re.search(r"\b(10|[1-9])\s*(?:st|nd|rd|th)?\b", tl) \
        or re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", tl)
    if m:
        g = m.group(1)
        return int(g) if g.isdigit() else _EN.get(g)
    return None


def _with_lang(item: dict, lang: Optional[str]) -> dict:
    out = dict(item)
    if lang in ("zh", "en"):
        out["lang"] = lang
    return out


def resolve_selection(thread_id: str, text: str) -> Optional[dict]:
    """把"回复编号 / 做第X个 / 做<菜名>"解析成具体候选 {cookId, name}。

    无近期候选 / 对不上 → None（调用方据此提示"先搜个菜谱"）。
    """
    items = recall_candidates(thread_id)
    if not items:
        return None
    lang = recall_candidate_lang(thread_id)
    t = (text or "").strip()

    n = _parse_ordinal(t)
    if n is not None and 1 <= n <= len(items):
        return _with_lang(items[n - 1], lang)

    compact = re.sub(r"\s+", "", t).lower().strip("，。！？,.!?")
    this_one = {
        "做这道", "做这个", "用设备做这道", "用设备做这个", "就这道", "就这个",
        "就这道吧", "就这个吧", "选这道", "选这个", "这个吧", "这道吧",
        "cookthisone", "makethisone", "cookit", "makeit",
    }
    if compact in this_one:
        if len(items) == 1:
            return _with_lang(items[0], lang)
        focus = recall_focus(thread_id)
        if focus and focus.get("verified_recipe"):
            focused_id = str(focus.get("recipe_id") or "")
            selected = next((item for item in items if str(item.get("cookId")) == focused_id), None)
            if selected:
                return _with_lang(selected, lang)

    for it in items:                    # 名字匹配（如"开始做一键红烧肉"）
        nm = it.get("name", "")
        if nm and (nm in t or t in nm):
            return _with_lang(it, lang)
    return None


def resolve_device_choice(text: str, devices: list):
    """从"1 / 第二台 / <设备名>"解析出 device_id；解析不到返回 None。devices=[(id, name), ...]。"""
    if not devices:
        return None
    t = (text or "").strip()
    # 设备序号必须是完整、明确的选择表达。不能复用菜品序号的宽松解析，
    # 否则“第一步做什么”会被误当成选择第 1 台设备。
    ordinal_match = re.fullmatch(
        r"\s*(?:(?:请|麻烦)?(?:选|选择|用|就用|我要用)\s*)?"
        r"(?:第\s*)?(10|[1-9]|[一二三四五六七八九十])\s*"
        r"(?:台|号)?\s*(?:设备|机器|锅)?\s*[。.!！]?\s*",
        t,
        flags=re.IGNORECASE,
    )
    n = None
    if ordinal_match:
        raw = ordinal_match.group(1)
        n = int(raw) if raw.isdigit() else _CN.get(raw)
    if n is not None and 1 <= n <= len(devices):
        return devices[n - 1][0]

    def normalized_name(value: str) -> str:
        value = re.sub(r"\s+", "", str(value or "").lower())
        value = value.strip("，。！？,.!?")
        value = re.sub(r"^(?:请|麻烦)?(?:用|选|选择|就用|我要用)", "", value)
        value = re.sub(r"^(?:please)?(?:use|select|choose|pick)(?:the)?", "", value)
        value = re.sub(r"(?:那一?台|这一?台|那台|这台|设备|机器|锅)$", "", value)
        value = re.sub(r"(?:device|machine)$", "", value)
        return value.strip("，。！？,.!?的")

    normalized_text = normalized_name(t)
    for did, nm in devices:
        direct_id = str(t or "").strip("，。！？,.!? ").lower() == str(did or "").strip().lower()
        if (nm and (nm in t or t in nm)) or direct_id:
            return did
        # 用户看到“办公室设备”后通常会说“用办公室那台”。去掉自然口语包装后
        # 再匹配一次，避免待选择状态被误清掉并落到 device_manage。
        normalized_device = normalized_name(nm)
        if (
            normalized_text
            and normalized_device
            and min(len(normalized_text), len(normalized_device)) >= 2
            and (normalized_text in normalized_device or normalized_device in normalized_text)
        ):
            return did
        # 英文界面展示的是本地化后的设备名称，设备状态里仍保留中文原名。
        # 接受用户照着界面回复的 "Use the Office Device"，但仍要求完整名称
        # 或明确别名，避免普通闲聊里的 office/showroom 被误识别为设备操作。
        for alias in _DEVICE_NAME_ALIASES_EN.get(str(nm or "").strip(), ()):
            normalized_alias = normalized_name(alias)
            if (
                normalized_text
                and normalized_alias
                and min(len(normalized_text), len(normalized_alias)) >= 3
                and normalized_text == normalized_alias
            ):
                return did
    return None


def set_pending(thread_id: str, cook_id: str, name: str,
                device_id: str = None, devices: list = None, lang: str = None,
                action_id: str | None = None, msg_id: int | None = None) -> None:
    """待确认开火态。devices 非空表示"还要选设备"；device_id 是已定/默认设备。"""
    if not action_id and msg_id is None:
        action_id, msg_id = new_device_action_identity()
    elif not action_id:
        action_id = new_device_action_identity()[0]
    elif msg_id is None:
        # 外部恢复旧 action 时没有 msgId，补一个并从此稳定复用。
        msg_id = new_device_action_identity()[1]
    normalized_msg_id = normalize_device_msg_id(msg_id)
    _pending[thread_id] = {
        "cookId": cook_id, "name": name,
        "device_id": device_id, "devices": devices or [],
        "action_id": str(action_id),
        "msg_id": normalized_msg_id,
        "lang": lang if lang in ("zh", "en") else None,
        "ts": time.time(),
    }
    set_pending_action(
        thread_id,
        "choose_device" if devices else "confirm_device_start",
        payload={
            "cookId": cook_id,
            "name": name,
            "action_id": str(action_id),
            "msg_id": normalized_msg_id,
        },
        lang=lang,
    )


def get_pending(thread_id: str, max_age: int = _PENDING_TTL) -> Optional[dict]:
    p = _pending.get(thread_id)
    if not p or time.time() - p["ts"] > max_age:
        if p:
            clear_pending(thread_id)
        return None
    return dict(p)


def clear_pending(thread_id: str) -> None:
    _pending.pop(thread_id, None)
    action = _pending_action.get(thread_id) or {}
    if action.get("kind") in {"choose_device", "confirm_device_start"}:
        _pending_action.pop(thread_id, None)


def consume_pending(
    thread_id: str,
    *,
    expected_action_id: str | None = None,
) -> Optional[dict]:
    """原子领取一次设备确认；并发确认只有一个调用方能拿到动作。"""
    pending = get_pending(thread_id)
    if not pending:
        return None
    if (
        expected_action_id
        and str(pending.get("action_id") or "") != str(expected_action_id)
    ):
        return None
    clear_pending(thread_id)
    return pending


def set_pending_action(
    thread_id: str,
    kind: str,
    *,
    payload: dict | None = None,
    lang: str | None = None,
) -> None:
    """记录助手下一轮明确等待的动作；不再从历史自然语言猜“嗯”的含义。"""
    if not thread_id or not str(kind or "").strip():
        return
    _pending_action[thread_id] = {
        "kind": str(kind).strip(),
        "payload": dict(payload or {}),
        "lang": lang if lang in ("zh", "en") else None,
        "ts": time.time(),
    }


def get_pending_action(thread_id: str, max_age: int = _PENDING_ACTION_TTL) -> Optional[dict]:
    action = _pending_action.get(thread_id)
    if not action:
        return None
    if time.time() - float(action.get("ts") or 0) > max_age:
        _pending_action.pop(thread_id, None)
        return None
    return {**action, "payload": dict(action.get("payload") or {})}


def clear_pending_action(thread_id: str) -> None:
    _pending_action.pop(thread_id, None)


def consume_pending_action(thread_id: str, expected_kind: str | None = None) -> Optional[dict]:
    action = get_pending_action(thread_id)
    if not action or (expected_kind and action.get("kind") != expected_kind):
        return None
    _pending_action.pop(thread_id, None)
    return action


def set_menu_task(thread_id: str, request: dict, search_result: dict, lang: str | None = None) -> None:
    if not thread_id or not isinstance(request, dict) or not isinstance(search_result, dict):
        return
    _menu_task[thread_id] = {
        "request": dict(request),
        "search_result": dict(search_result),
        "lang": lang if lang in ("zh", "en") else None,
        "ts": time.time(),
    }


def get_menu_task(thread_id: str, max_age: int = _PENDING_ACTION_TTL) -> Optional[dict]:
    task = _menu_task.get(thread_id)
    if not task:
        return None
    if time.time() - float(task.get("ts") or 0) > max_age:
        _menu_task.pop(thread_id, None)
        return None
    return dict(task)


def clear_menu_task(thread_id: str) -> None:
    _menu_task.pop(thread_id, None)


def set_active_cooking(thread_id: str, cook_id: str, name: str,
                       device_id: str = None, lang: str = None,
                       action_id: str | None = None, msg_id: int | None = None,
                       verification_status: str | None = None) -> None:
    """记录已确认启动的设备，后续"停止"优先发给同一台设备。"""
    _active[thread_id] = {
        "cookId": cook_id,
        "name": name,
        "device_id": device_id,
        "lang": lang if lang in ("zh", "en") else None,
        "action_id": str(action_id or "") or None,
        "msg_id": normalize_device_msg_id(msg_id) if msg_id is not None else None,
        "verification_status": str(verification_status or "running"),
        "ts": time.time(),
    }


def get_active_cooking(thread_id: str, max_age: int = _ACTIVE_TTL) -> Optional[dict]:
    rec = _active.get(thread_id)
    if not rec or time.time() - rec["ts"] > max_age:
        return None
    return rec


def clear_active_cooking(thread_id: str) -> None:
    _active.pop(thread_id, None)


def set_device_execution(thread_id: str, execution: dict) -> None:
    """覆盖当前设备执行工作副本；调用方只传结构化、脱敏字段。"""
    value = dict(execution or {})
    if not str(value.get("action_id") or "").strip():
        raise ValueError("device execution requires action_id")
    now = time.time()
    value["action_id"] = str(value["action_id"])
    if value.get("msg_id") is not None:
        value["msg_id"] = normalize_device_msg_id(value["msg_id"])
    value["updated_at"] = float(value.get("updated_at") or now)
    value["ts"] = float(value.get("ts") or value["updated_at"])
    _device_execution[thread_id] = value


def get_device_execution(
    thread_id: str,
    max_age: int = _EXECUTION_TTL,
) -> Optional[dict]:
    value = _device_execution.get(thread_id)
    if not value:
        return None
    if time.time() - float(value.get("updated_at") or value.get("ts") or 0) > max_age:
        _device_execution.pop(thread_id, None)
        return None
    return dict(value)


def update_device_execution(
    thread_id: str,
    *,
    expected_action_id: str,
    status: str,
    result_code: str | None = None,
    command_sent: bool | None = None,
    started: bool | None = None,
) -> bool:
    current = get_device_execution(thread_id)
    if not current or str(current.get("action_id") or "") != str(expected_action_id or ""):
        return False
    current.update(
        status=str(status),
        result_code=str(result_code or ""),
        command_sent=command_sent,
        started=started,
        updated_at=time.time(),
    )
    _device_execution[thread_id] = current
    return True


def clear_device_execution(thread_id: str) -> None:
    _device_execution.pop(thread_id, None)


def clear_thread(thread_id: str) -> None:
    """清除某个会话的短期编排状态；不会停止已经下发到设备的真实任务。"""
    _last.pop(thread_id, None)
    _pending.pop(thread_id, None)
    # 清聊天记忆不能假装停止真实设备任务；active 保留给后续“停止/进度”。
    _focus.pop(thread_id, None)
    _selected.pop(thread_id, None)
    _excluded.pop(thread_id, None)
    _search_clarification.pop(thread_id, None)
    _pending_action.pop(thread_id, None)
    _menu_task.pop(thread_id, None)
    _active_search.pop(thread_id, None)
    # 已领取的设备执行与 active_cooking 一样代表真实外部副作用，普通清会话不删。


def snapshot_thread_state(thread_id: str) -> ConversationTaskState:
    """导出当前线程的结构化快照，供 Redis ConversationMemory 持久化。"""
    candidate = recall_candidate_context(thread_id) or {}
    selected = get_selected_recipe(thread_id)
    excluded_ids = get_excluded_recipe_ids(thread_id)
    focus = recall_focus(thread_id)
    clarification = get_search_clarification(thread_id)
    pending_action = get_pending_action(thread_id)
    pending_device = get_pending(thread_id)
    menu_task = get_menu_task(thread_id)
    active = get_active_cooking(thread_id)
    device_execution = get_device_execution(thread_id)
    active_search = get_active_search_request(thread_id)

    if active:
        current_task = "device_cooking"
    elif pending_device:
        current_task = "device_start"
    elif isinstance(device_execution, dict) and device_execution.get("status") in {
        "dispatching", "submitted_unverified", "outcome_unknown",
    }:
        current_task = "device_execution"
    elif clarification:
        current_task = "recipe_search"
    elif menu_task:
        current_task = "menu_plan"
    elif candidate.get("items"):
        current_task = "recipe_selection"
    elif active_search:
        current_task = "recipe_search"
    elif focus:
        current_task = "recipe_discussion"
    else:
        current_task = None

    language = next(
        (
            value
            for value in (
                (pending_device or {}).get("lang"),
                (device_execution or {}).get("lang"),
                (selected or {}).get("lang"),
                candidate.get("lang"),
                (active_search or {}).get("lang"),
                (focus or {}).get("lang"),
                (active or {}).get("lang"),
            )
            if value in ("zh", "en")
        ),
        None,
    )
    timestamps = [
        float((value or {}).get("ts") or 0)
        for value in (
            candidate,
            active_search,
            selected,
            _excluded.get(thread_id),
            focus,
            clarification,
            pending_action,
            pending_device,
            device_execution,
            menu_task,
            active,
        )
    ]
    updated_at = max(timestamps, default=0.0)
    return ConversationTaskState(
        current_task=current_task,
        candidate_recipes=[dict(item) for item in candidate.get("items") or []],
        candidate_language=candidate.get("lang"),
        candidate_decision_id=str(candidate.get("decision_id") or ""),
        candidate_updated_at=(
            float(candidate.get("ts") or 0) or None
        ),
        active_search_request=dict(
            (active_search or {}).get("request")
            or candidate.get("search_request")
            or {}
        ),
        active_search_updated_at=(
            float((active_search or {}).get("ts") or 0) or None
        ),
        selected_recipe_id=(
            str((selected or {}).get("cookId") or "") or None
        ),
        selected_recipe_name=(
            str((selected or {}).get("name") or "") or None
        ),
        selected_recipe=(dict(selected) if selected else None),
        selected_recipe_updated_at=(
            float((selected or {}).get("ts") or 0) or None
        ),
        excluded_recipe_ids=excluded_ids,
        excluded_recipe_updated_at=(
            float((_excluded.get(thread_id) or {}).get("ts") or 0) or None
        ),
        focus=(dict(focus) if focus else None),
        pending_search_clarification=(
            dict(clarification) if clarification else None
        ),
        pending_action=(dict(pending_action) if pending_action else None),
        pending_device_start=(dict(pending_device) if pending_device else None),
        device_execution=(dict(device_execution) if device_execution else None),
        menu_task=(dict(menu_task) if menu_task else None),
        selected_device_id=str(
            (pending_device or {}).get("device_id")
            or (active or {}).get("device_id")
            or (device_execution or {}).get("device_id")
            or ""
        ) or None,
        active_cooking=(dict(active) if active else None),
        language=language,
        updated_at=updated_at,
    )


def restore_thread_state(
    thread_id: str,
    state: ConversationTaskState | dict | None,
) -> None:
    """用持久化快照覆盖指定线程缓存；不会影响其他用户线程。"""
    restored = (
        state
        if isinstance(state, ConversationTaskState)
        else ConversationTaskState.from_dict(state if isinstance(state, dict) else None)
    )
    for mapping in (
        _last,
        _pending,
        _active,
        _focus,
        _search_clarification,
        _pending_action,
        _menu_task,
        _selected,
        _excluded,
        _active_search,
        _device_execution,
    ):
        mapping.pop(thread_id, None)

    if restored.candidate_recipes:
        _last[thread_id] = {
            "items": [dict(item) for item in restored.candidate_recipes],
            "lang": restored.candidate_language,
            "decision_id": restored.candidate_decision_id,
            "search_request": dict(restored.active_search_request or {}),
            "ts": float(restored.candidate_updated_at or restored.updated_at or time.time()),
        }
    if restored.active_search_request:
        _active_search[thread_id] = {
            "request": dict(restored.active_search_request),
            "lang": restored.candidate_language or restored.language,
            # 旧 payload 没有专用字段时仅用顶层时间做一次迁移。
            "ts": float(
                restored.active_search_updated_at
                or restored.updated_at
                or time.time()
            ),
        }
    if restored.selected_recipe_id and restored.selected_recipe_name:
        selected = dict(restored.selected_recipe or {})
        selected.update({
            "cookId": restored.selected_recipe_id,
            "name": restored.selected_recipe_name,
            "ts": float(
                restored.selected_recipe_updated_at
                or restored.updated_at
                or time.time()
            ),
        })
        if restored.language in ("zh", "en") and not selected.get("lang"):
            selected["lang"] = restored.language
        _selected[thread_id] = selected
    if restored.excluded_recipe_ids:
        _excluded[thread_id] = {
            "ids": list(dict.fromkeys(
                str(item) for item in restored.excluded_recipe_ids if str(item)
            )),
            "ts": float(
                restored.excluded_recipe_updated_at
                or restored.updated_at
                or time.time()
            ),
        }
    for mapping, value in (
        (_focus, restored.focus),
        (_search_clarification, restored.pending_search_clarification),
        (_pending_action, restored.pending_action),
        (_pending, restored.pending_device_start),
        (_device_execution, restored.device_execution),
        (_menu_task, restored.menu_task),
        (_active, restored.active_cooking),
    ):
        if isinstance(value, dict):
            mapping[thread_id] = dict(value)


def task_state_has_data(state: ConversationTaskState | dict | None) -> bool:
    value = (
        state
        if isinstance(state, ConversationTaskState)
        else ConversationTaskState.from_dict(state if isinstance(state, dict) else None)
    )
    return value.has_data()


def routing_state_snapshot(
    thread_id: str,
    *,
    pending_clarification: dict | None = None,
) -> dict:
    """输出给路由器的最小进程内状态摘要，不暴露完整菜谱详情。"""
    candidate_context = recall_candidate_context(thread_id) or {}
    candidates = [
        {
            "cookId": str(item.get("cookId") or ""),
            "name": str(item.get("name") or ""),
        }
        for item in (candidate_context.get("items") or [])[:10]
        if str(item.get("cookId") or "").strip()
    ]
    focus = recall_focus(thread_id)
    selected = get_selected_recipe(thread_id)
    active_search = get_active_search_request(thread_id)
    active_request = dict((active_search or {}).get("request") or {})
    active_constraints = {
        key: [
            str(item)[:40]
            for item in value[:8]
            if str(item).strip()
        ]
        for key, value in active_request.items()
        if key in {
            "ingredients", "cuisines", "flavors", "scenes", "exclude",
            "exclude_cuisines", "avoid", "methods", "meals",
            "dietary_constraints",
        }
        and isinstance(value, list)
        and value
    }
    return {
        "pending_action": get_pending_action(thread_id),
        "pending_device_start": get_pending(thread_id),
        "device_execution": get_device_execution(thread_id),
        "active_cooking": get_active_cooking(thread_id),
        "pending_clarification": (
            dict(pending_clarification)
            if isinstance(pending_clarification, dict)
            else get_search_clarification(thread_id)
        ),
        "latest_candidates": candidates,
        "latest_focus": (
            {
                "name": str(focus.get("name") or focus.get("topic") or ""),
                "verified_recipe": bool(focus.get("verified_recipe")),
                "source": str(focus.get("source") or ""),
            }
            if focus else None
        ),
        "selected_recipe": (
            {
                "cookId": str(selected.get("cookId") or ""),
                "name": str(selected.get("name") or ""),
            }
            if selected else None
        ),
        "excluded_recipe_ids": get_excluded_recipe_ids(thread_id),
        "current_task": snapshot_thread_state(thread_id).current_task,
        "active_constraints": active_constraints,
    }


_CONFIRM_EXACT = {
    "确认", "确认开始", "确认开火", "开始烹饪",
    "confirm", "confirm start", "confirm cooking", "start cooking",
}
_CANCEL_EXACT = {"取消", "算了", "不", "不要", "不用", "别", "no", "n", "cancel", "stop", "停", "先不"}
_CANCEL_SUB = ("取消", "算了", "不做", "别做", "不想做", "先不做", "cancel")

_VOICE_FILLERS = {
    "嗯", "嗯嗯", "啊", "呃", "额", "那个", "好的", "好", "行", "可以", "吧",
    "uh", "um", "okay", "ok", "please",
}
_VOICE_COMMANDS = {
    # 确认命令仍只映射到 is_confirm 已允许的精确短语，不能放宽成包含判断。
    "确认": "确认",
    "确认开始": "确认开始",
    "确认开火": "确认开火",
    "开始烹饪": "开始烹饪",
    # QQ ASR 偶尔把“开始烹饪”重复为“烹饪，开始烹饪”。
    "烹饪开始烹饪": "开始烹饪",
    "confirm": "confirm",
    "confirmstart": "confirm start",
    "confirmcooking": "confirm cooking",
    "startcooking": "start cooking",
    # 取消/停止只做同样的精确归一化，不从长句中截取控制词。
    "取消": "取消",
    "算了": "算了",
    "不要": "不要",
    "不用": "不用",
    "先不": "先不",
    "不做": "不做",
    "先不做": "先不做",
    "cancel": "cancel",
    "停止": "停止",
    "停止烹饪": "停止烹饪",
    "停": "停",
    "暂停": "暂停",
    "别做了": "别做了",
    "stop": "stop",
    "stopcooking": "stop cooking",
    "pause": "pause",
}


def normalize_voice_control_text(text: str) -> str:
    """保守清理 QQ ASR 控制语句；非精确命令只去展示前缀，不改变原意。"""
    raw = re.sub(r"^\s*\[voice\]\s*", "", str(text or ""), flags=re.IGNORECASE).strip()
    if not raw:
        return ""

    # 仅从首尾移除无语义口头词；句子中出现其它内容时不会被识别成控制命令。
    tokens = [part.strip().lower() for part in re.split(r"[，,。.!！?？；;、\s]+", raw) if part.strip()]
    while tokens and tokens[0] in _VOICE_FILLERS:
        tokens.pop(0)
    while tokens and tokens[-1] in _VOICE_FILLERS:
        tokens.pop()
    compact = "".join(tokens)
    return _VOICE_COMMANDS.get(compact, raw)


def is_cancel(text: str) -> bool:
    """待确认态下，用户是否在取消（先判 cancel，避免"不确认"这类误开火）。"""
    t = (text or "").strip().lower()
    if t in _CANCEL_EXACT:
        return True
    if t[:1] in ("不", "别") or t.startswith("先不"):
        return True
    return any(w in t for w in _CANCEL_SUB)


def is_confirm(text: str) -> bool:
    """待确认态下是否给出无歧义的开火确认；只接受完整短语。"""
    t = re.sub(r"[，,。.!！?？]+$", "", (text or "").strip().lower()).strip()
    return t in _CONFIRM_EXACT


_STOP_EXACT = {"停", "停止", "停一下", "别做了", "不做了", "暂停", "取消烹饪", "停止烹饪",
               "stop", "pause", "stop cooking", "cancel cooking"}
_ABANDON_EXACT = {
    "算了", "算了吧", "好吧算了", "我放弃", "好吧我放弃", "不弄了",
    "不继续了", "先这样吧", "到此为止", "never mind", "i give up",
    "forget it", "leave it",
}


def is_abandonment(text: str) -> bool:
    """识别取消当前对话任务的口头语，但不把它直接解释成停止设备。"""
    value = re.sub(r"[，,。.!！?？；;\s]+", "", str(text or "").strip().lower())
    normalized = {
        re.sub(r"[，,。.!！?？；;\s]+", "", item.lower())
        for item in _ABANDON_EXACT
    }
    return value in normalized


def is_stop(text: str) -> bool:
    """是否要停止当前烹饪（recipe_execute 里的停止子意图）。"""
    t = (text or "").strip().lower()
    return t in _STOP_EXACT or "停止" in t or "别做了" in t or "暂停" in t or "stop" in t


def is_progress(text: str) -> bool:
    """是否在查询当前烹饪进度，而不是搜索一道名为“做到哪里”的菜。"""
    t = (text or "").strip().lower()
    return any(marker in t for marker in (
        "做到哪里", "做到哪", "哪一步", "进行到", "进度", "还要多久", "多久做好",
        "cooking progress", "how long", "which step", "how far",
    ))
