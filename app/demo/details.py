"""演示排期所需的详情水合。

真实步骤存在时原样保留；公开展示无法携带生产详情时，只注入明确标记的
虚拟模板。模板用于演示资源编排，不是食谱事实，也不能用于真实设备执行。
"""

from __future__ import annotations

import os

from .models import Brief, Recipe


def virtual_schedule_enabled() -> bool:
    return os.getenv("DEMO_VIRTUAL_SCHEDULE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _template_equipment(
    recipe: Recipe,
    brief: Brief,
    *,
    dish_index: int,
) -> str:
    equipment = list(dict.fromkeys(brief.equipment))
    if recipe.kind == "soup" and "灶台" in equipment:
        return "灶台"
    if "空气炸锅" in equipment and (
        any(marker in recipe.name for marker in ("烤", "炸")) or dish_index == 0
    ):
        return "空气炸锅"
    if "灶台" in equipment:
        return "灶台"
    return equipment[0] if equipment else "通用厨具"


def hydrate_schedule_details(
    brief: Brief,
    candidates: list[Recipe],
    recipe_ids: list[str],
) -> tuple[list[Recipe], dict]:
    """只水合最终菜单；返回的新对象可安全写入 Graph 状态。"""
    selected = set(recipe_ids)
    enabled = virtual_schedule_enabled()
    hydrated: list[Recipe] = []
    source_count = 0
    template_count = 0
    missing_count = 0
    dish_index = 0
    for recipe in candidates:
        if recipe.id not in selected:
            hydrated.append(recipe)
            continue
        if recipe.steps:
            if recipe.detail_basis == "demo_template":
                template_count += 1
                hydrated.append(recipe)
            else:
                source_count += 1
                hydrated.append(recipe.model_copy(update={"detail_basis": "source"}))
            if recipe.kind == "dish":
                dish_index += 1
            continue
        if not enabled:
            missing_count += 1
            hydrated.append(recipe.model_copy(update={"detail_basis": "missing"}))
            if recipe.kind == "dish":
                dish_index += 1
            continue

        equipment = _template_equipment(recipe, brief, dish_index=dish_index)
        if recipe.kind == "dish":
            dish_index += 1
        cook_minutes = 20 if recipe.kind == "soup" else (
            16 if equipment == "空气炸锅" else 12
        )
        steps = [
            "演示模板：食材准备 5 分钟（不来自生产食谱详情）",
            f"演示模板：使用{equipment}烹饪 {cook_minutes} 分钟（仅用于排期展示）",
        ]
        template_count += 1
        hydrated.append(recipe.model_copy(update={
            "steps": steps,
            "detail_basis": "demo_template",
        }))
    return hydrated, {
        "source_count": source_count,
        "demo_template_count": template_count,
        "missing_count": missing_count,
        "notice": (
            "生产食谱步骤未包含在公开展示数据中；虚拟步骤只用于演示排期，"
            "不作为真实做法或设备执行依据。"
            if template_count
            else "排期仅使用来源中已有的步骤信息。"
        ),
    }
