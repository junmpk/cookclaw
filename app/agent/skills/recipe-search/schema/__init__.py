"""标准菜谱数据模型（语言无关骨架 + i18n 文本）。

见知识库《CookClaw 标准菜谱数据模型与多语言检索架构》。
"""
from .recipe_schema import (
    Facets,
    I18nText,
    Ingredient,
    NutritionFacts,
    Recipe,
    Step,
    StepDeviceParams,
    load_facet_labels,
)

__all__ = [
    "Recipe",
    "Facets",
    "Ingredient",
    "Step",
    "StepDeviceParams",
    "NutritionFacts",
    "I18nText",
    "load_facet_labels",
]
