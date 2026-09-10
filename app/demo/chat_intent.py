"""多 Agent 对话桥的确定性触发、确认和局部修订解析。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Brief

_GRAPH_REQUEST = re.compile(
    r"(?:三位专家|三个专家|多\s*agent|multi[- ]?agent|langgraph|"
    r"协作配餐|协作规划|专家规划)",
    flags=re.IGNORECASE,
)
_GRAPH_CONFIRM = re.compile(
    r"^(?:确认|同意|开始|执行|好的|可以|confirm|yes|approve)$",
    re.IGNORECASE,
)
_GRAPH_CANCEL = re.compile(
    r"^(?:取消|不要|先不做|算了|cancel|no|reject)$",
    re.IGNORECASE,
)
_SLOT = re.compile(
    r"(?:第\s*([一二两三四五六七八九十\d]+)\s*(?:道|个)?|"
    r"(?:option|dish|item)\s*([1-7]))",
    re.IGNORECASE,
)
_ADJUSTMENT = re.compile(
    r"(?:换掉|替换|换一下|换成|调整|改成|不要(?:这)?道|"
    r"清淡|少油|少盐|不辣|不要辣|replace|change|lighter|less\s+(?:oil|salt))",
    re.IGNORECASE,
)
_UNSPECIFIED_ADJUSTMENT = re.compile(
    r"(?:这道|这一个|这个菜|某道).*(?:换|替换|调整|不要)|"
    r"(?:换|替换|调整).*(?:这道|这一个|这个菜|某道)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SlotAdjustment:
    slot_index: int
    request: str


def requests_graph(text: str) -> bool:
    return bool(_GRAPH_REQUEST.search(str(text or "")))


def decision_from_chat(text: str) -> bool | None:
    value = re.sub(r"\s+", "", str(text or ""))
    if _GRAPH_CONFIRM.fullmatch(value):
        return True
    if _GRAPH_CANCEL.fullmatch(value):
        return False
    return None


def slot_adjustment_from_chat(text: str) -> SlotAdjustment | None:
    value = str(text or "").strip()
    slot = _SLOT.search(value)
    if not slot or not _ADJUSTMENT.search(value):
        return None
    raw = slot.group(1) or slot.group(2)
    number = int(raw) if raw.isdigit() else {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }.get(raw, 0)
    if not 1 <= number <= 7:
        return None
    return SlotAdjustment(slot_index=number - 1, request=value[:300])


def requests_unspecified_adjustment(text: str) -> bool:
    return bool(_UNSPECIFIED_ADJUSTMENT.search(str(text or "")))


def brief_from_task_state(state, *, mode: str = "live") -> Brief | None:
    """从共享 Runtime 已保存的任务状态构造 Graph Brief。"""
    menu = state.menu_task or {}
    request = dict(menu.get("request") or state.active_search_request or {})
    dishes = int(request.get("menu_dish_count") or 0)
    soups = int(request.get("menu_soup_count") or 0)
    if dishes <= 0 or soups <= 0:
        return None
    exclusions = list(request.get("exclude") or [])
    exclusions.extend(
        item for item in request.get("avoid") or [] if item not in exclusions
    )
    return Brief(
        request=str(
            request.get("original_text") or request.get("query") or ""
        ).strip(),
        people=int(request.get("party_size") or 3),
        dishes=dishes,
        soups=soups,
        exclusions=[
            str(item) for item in exclusions if str(item).strip()
        ][:12],
        mode=mode if mode in {"live", "rehearsal"} else "live",
    )
