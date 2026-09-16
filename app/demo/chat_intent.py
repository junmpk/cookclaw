"""多 Agent 对话桥的确定性触发、确认和局部修订解析。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Brief, ComplexityDimensions, ComplexityProfile

_GRAPH_REQUEST = re.compile(
    r"(?:三位专家|三个专家|多\s*agent|multi[- ]?agent|langgraph|"
    r"协作配餐|协作规划|专家规划)",
    flags=re.IGNORECASE,
)
_MENU_PLAN_HINT = re.compile(
    r"(?:安排|规划|准备|设计|搭配).{0,12}(?:餐|菜单|菜|汤)|"
    r"[一二两三四五六七八九十\d]+\s*菜.{0,8}[零一二两三\d]+\s*汤|"
    r"(?:family|dinner|meal|menu).{0,24}(?:plan|planning)|"
    r"(?:plan|planning).{0,24}(?:family|dinner|meal|menu)",
    flags=re.IGNORECASE,
)
_INVENTORY_HINT = re.compile(
    r"(?:冰箱|现有食材|已有食材|家里(?:有|现有)|手头(?:有|现有)|"
    r"尽量(?:使用|利用)|减少浪费|采购|购物清单|预算|"
    r"fridge|pantry|inventory|available ingredients|shopping|budget)",
    flags=re.IGNORECASE,
)
_SCHEDULING_HINT = re.compile(
    r"(?:同时(?:做好|出锅)|按时出锅|分钟内|小时内|灶台|烤箱|蒸箱|"
    r"空气炸锅|电饭煲|设备有限|先后顺序|并行|排期|"
    r"ready at|within .{0,8}(?:minutes?|hours?)|stove|oven|air fryer|schedule)",
    flags=re.IGNORECASE,
)
_DIETARY_HINT = re.compile(
    r"(?:过敏|忌口|低盐|少盐|低脂|控糖|素食|清淡|健康目标|"
    r"allerg|diet|low[- ]?(?:salt|fat|sugar)|vegetarian)",
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


@dataclass(frozen=True)
class GraphRoutingDecision:
    """可展示、可测试的多 Agent 路由决定。"""

    use_graph: bool
    score: int
    trigger: str
    reasons: tuple[str, ...]
    profile: ComplexityProfile | None = None

    def public_data(self) -> dict:
        return {
            "use_graph": self.use_graph,
            "score": self.score,
            "trigger": self.trigger,
            "reasons": list(self.reasons),
            "profile": self.profile.model_dump() if self.profile else None,
        }


def requests_graph(text: str) -> bool:
    return bool(_GRAPH_REQUEST.search(str(text or "")))


def might_need_graph(text: str) -> bool:
    """IM 通道的廉价预筛；最终决定仍以结构化 Brief 为准。"""
    value = str(text or "")
    return requests_graph(value) or bool(_MENU_PLAN_HINT.search(value))


def graph_routing_decision(
    text: str,
    brief: Brief | None,
) -> GraphRoutingDecision:
    """根据共享 Runtime 的结构化结果决定是否进入多 Agent 子图。

    显式关键词只作为调试/演示覆盖。自动路由依据菜单组合复杂度、人数和硬约束，
    避免再调用一个不可解释且可能抖动的 LLM 分类器。
    """
    explicit = requests_graph(text)
    if brief is None:
        return GraphRoutingDecision(
            use_graph=False,
            score=0,
            trigger="insufficient_structure",
            reasons=("缺少可执行的多菜菜单结构",),
        )

    menu_score = 0
    dietary_score = 0
    inventory_score = 0
    scheduling_score = 0
    reasons: list[str] = []
    total = int(brief.dishes) + int(brief.soups)
    if brief.dishes > 0 and brief.soups > 0:
        menu_score += 2
        reasons.append("需要同时组合菜和汤")
    if total >= 3:
        menu_score += 2
        reasons.append(f"需要规划 {total} 个菜单槽位")
    elif total >= 2:
        menu_score += 1
        reasons.append(f"需要规划 {total} 个菜单槽位")
    if brief.people >= 4:
        menu_score = min(4, menu_score + 1)
        reasons.append(f"需要服务 {brief.people} 人")
    elif brief.people >= 2:
        reasons.append(f"需要服务 {brief.people} 人")

    request_text = f"{text} {brief.request}"
    if brief.exclusions or _DIETARY_HINT.search(request_text):
        dietary_score = 2 if brief.exclusions else 1
        reasons.append(
            "存在必须排除的食材"
            if brief.exclusions
            else "存在饮食或健康偏好"
        )
    if brief.available_ingredients or _INVENTORY_HINT.search(request_text):
        inventory_score = 2
        reasons.append("需要核对现有食材与采购缺口")
    if brief.budget_yuan:
        inventory_score = min(3, inventory_score + 1)
        reasons.append(f"采购预算不超过 {brief.budget_yuan} 元")
    if total >= 4:
        # 菜多只表示“排期有价值”，不能单独触发 Scheduler；还需要明确
        # 时限/并行诉求，或与有限设备约束组合后才达到选择阈值。
        scheduling_score += 1
        reasons.append("多道菜存在制作顺序优化空间")
    if brief.max_minutes or _SCHEDULING_HINT.search(request_text):
        scheduling_score += 2
        reasons.append(
            f"需要在 {brief.max_minutes} 分钟内完成"
            if brief.max_minutes
            else "存在明确的出餐时间要求"
        )
    if brief.equipment:
        scheduling_score = min(4, scheduling_score + 1)
        reasons.append("需要协调有限的厨房设备")

    dimensions = ComplexityDimensions(
        menu=min(4, menu_score),
        dietary=min(3, dietary_score),
        inventory=min(3, inventory_score),
        scheduling=min(4, scheduling_score),
    )
    score = sum(dimensions.model_dump().values())
    use_graph = explicit or score >= 2
    if explicit:
        reasons.insert(0, "用户显式要求多 Agent 协作")
    selected_agents = []
    if use_graph:
        selected_agents = ["research", "dietary"]
        if dimensions.inventory >= 2:
            selected_agents.append("inventory")
        selected_agents.append("menu")
        if dimensions.scheduling >= 2:
            selected_agents.append("scheduler")
    profile = ComplexityProfile(
        level=(
            "advanced"
            if any(name in selected_agents for name in ("inventory", "scheduler"))
            else ("coordinated" if use_graph else "simple")
        ),
        total_score=score,
        dimensions=dimensions,
        selected_agents=selected_agents,
        reasons=list(dict.fromkeys(reasons)),
    )
    return GraphRoutingDecision(
        use_graph=use_graph,
        score=score,
        trigger="explicit" if explicit else ("complexity" if use_graph else "simple"),
        reasons=tuple(profile.reasons),
        profile=profile,
    )


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


def brief_from_search_request(request_value, *, mode: str = "live") -> Brief | None:
    """从共享 Router 的结构化 SearchRequest 构造 Graph Brief。"""
    if hasattr(request_value, "model_dump"):
        request = dict(request_value.model_dump())
    else:
        request = dict(request_value or {})
    dishes = int(request.get("menu_dish_count") or 0)
    soups = int(request.get("menu_soup_count") or 0)
    if dishes <= 0 or soups < 0 or dishes + soups < 2:
        return None
    exclusions = list(request.get("exclude") or [])
    exclusions.extend(
        item for item in request.get("avoid") or [] if item not in exclusions
    )
    original_text = str(
        request.get("original_text") or request.get("query") or ""
    ).strip()
    available = list(
        request.get("available_ingredients")
        or request.get("required_ingredients")
        or request.get("ingredients")
        or []
    )
    minute_match = re.search(r"(\d{1,4})\s*分钟", original_text)
    hour_match = re.search(r"(\d{1,2})\s*小时", original_text)
    max_minutes = (
        int(minute_match.group(1))
        if minute_match
        else (int(hour_match.group(1)) * 60 if hour_match else None)
    )
    if max_minutes is None and "半小时" in original_text:
        max_minutes = 30
    budget_match = re.search(r"(\d{1,4})\s*元(?:以内|以下|预算)?", original_text)
    equipment = [
        name
        for name in ("灶台", "烤箱", "蒸箱", "空气炸锅", "电饭煲")
        if name in original_text
    ]
    return Brief(
        request=original_text,
        people=int(request.get("party_size") or 3),
        dishes=dishes,
        soups=soups,
        exclusions=[
            str(item) for item in exclusions if str(item).strip()
        ][:12],
        available_ingredients=[
            str(item) for item in available if str(item).strip()
        ][:20],
        budget_yuan=int(budget_match.group(1)) if budget_match else None,
        max_minutes=max_minutes,
        equipment=equipment,
        mode=mode if mode in {"live", "rehearsal"} else "live",
    )


def brief_from_task_state(state, *, mode: str = "live") -> Brief | None:
    """从共享 Runtime 已保存的任务状态构造 Graph Brief。"""
    menu = state.menu_task or {}
    request = dict(menu.get("request") or state.active_search_request or {})
    return brief_from_search_request(request, mode=mode)
