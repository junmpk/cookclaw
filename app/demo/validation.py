"""普通 Python 业务规则：不让 LLM 决定候选真实性和硬约束。"""
from .models import Brief, Recipe

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
