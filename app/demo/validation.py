"""普通 Python 业务规则：不让 LLM 决定候选真实性和硬约束。"""
from .models import Brief, PlanValidationReport, Recipe

ALIASES = {
    "花生": ("花生", "peanut", "groundnut"),
    "鸡蛋": ("鸡蛋", "egg"),
    "牛奶": ("牛奶", "milk", "cheese", "butter", "cream", "yogurt"),
    "小麦": ("小麦", "wheat", "flour", "pasta", "noodle"),
    "大豆": ("大豆", "soy", "tofu"),
    "鱼": ("鱼", "fish", "salmon", "tuna", "cod"),
    "虾": ("虾", "shrimp", "prawn"),
}


def conflicts(recipe: Recipe, exclusions: list[str]) -> list[str]:
    text = " ".join(recipe.ingredients).casefold()
    return [term for term in exclusions if any(
        alias.casefold() in text for alias in ALIASES.get(term, (term,))
    )]


def validate_menu(brief: Brief, candidates: list[Recipe], ids: list[str]) -> list[str]:
    catalog = {r.id: r for r in candidates}
    issues = []
    if len(ids) != len(set(ids)):
        issues.append("菜单包含重复菜品")
    if any(item not in catalog for item in ids):
        issues.append("菜单含检索候选之外的 ID")
    selected = [catalog[item] for item in ids if item in catalog]
    if sum(r.kind == "dish" for r in selected) != brief.dishes:
        issues.append(f"需要 {brief.dishes} 道菜")
    if sum(r.kind == "soup" for r in selected) != brief.soups:
        issues.append(f"需要 {brief.soups} 道汤")
    for recipe in selected:
        blocked = conflicts(recipe, brief.exclusions)
        if blocked:
            issues.append(f"{recipe.name} 含排除食材：{', '.join(blocked)}")
        if not recipe.ingredients:
            issues.append(f"{recipe.name} 缺少食材证据")
    return issues


def validate_plan(
    brief: Brief,
    candidates: list[Recipe],
    menu: dict,
    *,
    dietary: dict | None = None,
    inventory: dict | None = None,
    schedule: dict | None = None,
    review: dict | None = None,
) -> PlanValidationReport:
    """在人工确认前汇总所有 Agent 输出，并检查跨模块一致性。"""
    menu_ids = [str(item) for item in menu.get("recipe_ids") or []]
    blocking = validate_menu(brief, candidates, menu_ids)
    warnings: list[str] = []
    checks = {
        "menu": "failed" if blocking else "passed",
        "dietary": "not_applicable",
        "inventory": "not_applicable",
        "schedule": "not_applicable",
        "budget": "not_applicable",
        "review": "not_applicable",
    }

    if dietary is not None:
        checks["dietary"] = "passed"
        warnings.extend(str(item) for item in dietary.get("unknowns") or [])

    if inventory is not None:
        preferred = {
            str(item) for item in inventory.get("preferred_recipe_ids") or []
        }
        candidate_ids = {recipe.id for recipe in candidates}
        if not preferred.issubset(candidate_ids):
            blocking.append("库存分析引用了候选之外的食谱")
            checks["inventory"] = "failed"
        else:
            checks["inventory"] = "passed"
        warnings.extend(str(item) for item in inventory.get("unknowns") or [])
        if brief.budget_yuan:
            checks["budget"] = "warning"
            warnings.append(
                f"缺少实时价格，无法证明采购金额不超过 {brief.budget_yuan} 元。"
            )

    if schedule is not None:
        tasks = list(schedule.get("tasks") or [])
        task_ids = [str(item.get("recipe_id") or "") for item in tasks]
        if len(task_ids) != len(menu_ids) or set(task_ids) != set(menu_ids):
            blocking.append("烹饪排期没有完整覆盖最终菜单")
            checks["schedule"] = "failed"
        elif schedule.get("deadline_status") == "needs_confirmation":
            checks["schedule"] = "warning"
        else:
            checks["schedule"] = "passed"
        warnings.extend(str(item) for item in schedule.get("unknowns") or [])
        if any(item.get("evidence_basis") == "demo_template" for item in tasks):
            warnings.append(
                "当前时间与设备安排包含公开展示用虚拟步骤，不能作为真实做法或设备指令。"
            )
        total_minutes = schedule.get("total_minutes")
        if (
            brief.max_minutes is not None
            and isinstance(total_minutes, int)
            and schedule.get("deadline_status") == "verifiable"
            and total_minutes > brief.max_minutes
        ):
            blocking.append(
                f"当前可核验排期为 {total_minutes} 分钟，超过要求的 {brief.max_minutes} 分钟"
            )
            checks["schedule"] = "failed"

    if review is not None:
        if review.get("needs_revision"):
            blocking.extend(str(item) for item in review.get("findings") or [])
            checks["review"] = "failed"
        else:
            checks["review"] = "passed"
        warnings.extend(str(item) for item in review.get("unknowns") or [])

    blocking = list(dict.fromkeys(blocking))[:12]
    warnings = list(dict.fromkeys(warnings))[:12]
    return PlanValidationReport(
        status=(
            "blocked"
            if blocking
            else ("needs_confirmation" if warnings else "ready")
        ),
        blocking_issues=blocking,
        warnings=warnings,
        checks=checks,
    )
