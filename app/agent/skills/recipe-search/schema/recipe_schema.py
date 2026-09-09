"""
标准菜谱数据模型（Pydantic v2）—— 语言无关骨架 + i18n 本地化文本。

设计见知识库《CookClaw 标准菜谱数据模型与多语言检索架构》。要点：
  - 语言无关骨架：facets（canonical slug + 数值）/ 结构化食材 / 可执行步骤参数
  - 语言相关文本：i18n（每语言一份 name/description/食材名/步骤文案/展示标签）
  - 分类维度取值受 facet_labels.json（受控词表）约束；数值维度带范围

用法：
  .venv/bin/python app/agent/skills/recipe-search/schema/recipe_schema.py
    → 导出 recipe.schema.json + 跑正/负例自测
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

_SCHEMA_DIR = Path(__file__).resolve().parent
_LABELS_PATH = _SCHEMA_DIR / "facet_labels.json"


def load_facet_labels() -> dict:
    """加载受控词表（canonical slug → 多语言 label）。"""
    return json.loads(_LABELS_PATH.read_text(encoding="utf-8"))


_LABELS = load_facet_labels()
# 各分类维度的合法 slug 集合（以 `_` 开头的是元数据，非维度）
_VOCAB: Dict[str, set] = {
    dim: set(vals.keys()) for dim, vals in _LABELS.items() if not dim.startswith("_")
}

Lang = Literal["zh", "en"]
# 受词表约束的分类维度（与 Facets 上的 list 字段一一对应）
_CATEGORICAL_DIMS = [
    "cuisine", "main_ingredient", "flavor", "method", "scene", "meal",
    "nutrition", "diet", "allergens", "device", "cookware",
]


class NutritionFacts(BaseModel):
    protein_g: Optional[float] = Field(None, ge=0)
    fat_g: Optional[float] = Field(None, ge=0)
    carb_g: Optional[float] = Field(None, ge=0)
    fiber_g: Optional[float] = Field(None, ge=0)


class Facets(BaseModel):
    """语言无关的筛选维度。分类维度取值须在 facet_labels.json 词表内。"""
    # 分类（canonical slug）
    cuisine: List[str] = []
    main_ingredient: List[str] = []
    flavor: List[str] = []
    method: List[str] = []
    scene: List[str] = []
    meal: List[str] = []
    nutrition: List[str] = []
    # 约束（硬过滤）
    diet: List[str] = []
    allergens: List[str] = []
    # 设备 / 锅具
    device: List[str] = []
    cookware: List[str] = []
    # 数值 / 分级（支持范围筛）
    difficulty: Optional[int] = Field(None, ge=1, le=5)
    total_time_min: Optional[int] = Field(None, ge=0)
    prep_time_min: Optional[int] = Field(None, ge=0)
    cook_time_min: Optional[int] = Field(None, ge=0)
    spiciness: Optional[int] = Field(None, ge=0, le=5)
    servings: Optional[int] = Field(None, ge=1)
    calorie_kcal: Optional[int] = Field(None, ge=0)
    nutrition_facts: Optional[NutritionFacts] = None

    def unknown_values(self) -> Dict[str, List[str]]:
        """返回不在受控词表内的分类取值（软校验，供灌库脚本决策；不 raise）。"""
        out: Dict[str, List[str]] = {}
        for dim in _CATEGORICAL_DIMS:
            vocab = _VOCAB.get(dim, set())
            bad = [v for v in getattr(self, dim) if v not in vocab]
            if bad:
                out[dim] = bad
        return out


class Ingredient(BaseModel):
    """结构化食材：用量 + 单位 + 过敏原（name_key 语言无关，本地名进 i18n）。"""
    name_key: str
    amount: Optional[float] = Field(None, ge=0)
    unit: Optional[str] = None
    group: str = "main"             # main / season / garnish / ...
    optional: bool = False
    allergen: Optional[str] = None  # ∈ allergens 词表


class StepDeviceParams(BaseModel):
    """设备可执行参数（对接 recipe-operation 的执行指令）。"""
    temp_c: Optional[float] = None
    speed: Optional[int] = Field(None, ge=0)
    duration_s: Optional[int] = Field(None, ge=0)


class Step(BaseModel):
    order: int = Field(ge=1)
    text_key: str
    action: Optional[str] = None             # 受控动作集（待与设备指令映射）
    device_params: Optional[StepDeviceParams] = None


class I18nText(BaseModel):
    """某一语言的展示文本。"""
    name: str
    description: str = ""
    ingredient_names: Dict[str, str] = {}    # name_key -> 本地名
    step_texts: List[str] = []
    display_tags: List[str] = []


class Recipe(BaseModel):
    """标准菜谱：语言无关骨架 + 每语言一份 i18n 文本。"""
    recipe_id: str
    source: str = "unknown"
    image_url: Optional[str] = None
    canonical_lang: Lang = "zh"
    version: int = 1
    updated_at: Optional[str] = None

    facets: Facets = Field(default_factory=Facets)
    ingredients: List[Ingredient] = []
    steps: List[Step] = []
    i18n: Dict[str, I18nText]

    @model_validator(mode="after")
    def _check_consistency(self):
        # 1) 源语言必须有对应 i18n 文本
        if self.canonical_lang not in self.i18n:
            raise ValueError(
                f"canonical_lang={self.canonical_lang!r} 在 i18n 中缺失（i18n 语言：{list(self.i18n)}）"
            )
        # 2) i18n 的 ingredient_names 只能引用已声明的食材 name_key
        name_keys = {ing.name_key for ing in self.ingredients}
        for lang, txt in self.i18n.items():
            unknown = set(txt.ingredient_names) - name_keys
            if unknown:
                raise ValueError(f"i18n[{lang}].ingredient_names 引用了未声明的食材：{sorted(unknown)}")
        # 3) 步骤 order 必须唯一
        orders = [s.order for s in self.steps]
        if len(orders) != len(set(orders)):
            raise ValueError(f"steps.order 存在重复：{orders}")
        return self

    def unknown_facets(self) -> Dict[str, List[str]]:
        """软校验入口：facets 中越出词表的取值（供灌库时告警 / 补词表）。"""
        return self.facets.unknown_values()


if __name__ == "__main__":
    # 1) 导出 JSON Schema
    out = _SCHEMA_DIR / "recipe.schema.json"
    out.write_text(json.dumps(Recipe.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ JSON Schema 导出 → {out.name}（{len(_VOCAB)} 个受控维度）")

    # 2) 正例：构造一条标准菜谱（红烧肉）
    sample = Recipe(
        recipe_id="1369222630114693122",
        source="external",
        canonical_lang="zh",
        image_url="https://images.example.invalid/cook_platform/images/x.jpg",
        facets=Facets(
            cuisine=["sichuan"], main_ingredient=["pork"], flavor=["savory"],
            method=["red_braised"], meal=["main_course"],
            difficulty=2, total_time_min=60, spiciness=1, servings=3,
            calorie_kcal=520, allergens=["soy"], device=["cookclaw_k3501"], cookware=["wok"],
            nutrition_facts=NutritionFacts(protein_g=22, fat_g=38, carb_g=12),
        ),
        ingredients=[
            Ingredient(name_key="pork_belly", amount=500, unit="g", group="main"),
            Ingredient(name_key="soy_sauce", amount=30, unit="ml", group="season", allergen="soy"),
        ],
        steps=[
            Step(order=1, text_key="s1", action="sear",
                 device_params=StepDeviceParams(temp_c=180, speed=1, duration_s=300)),
            Step(order=2, text_key="s2", action="braise",
                 device_params=StepDeviceParams(temp_c=105, speed=0, duration_s=2400)),
        ],
        i18n={
            "zh": I18nText(name="稻香红烧肉", description="肥而不腻",
                           ingredient_names={"pork_belly": "五花肉", "soy_sauce": "生抽"},
                           step_texts=["大火煎香五花肉", "小火慢炖40分钟"],
                           display_tags=["川菜", "家常菜", "咸鲜"]),
            "en": I18nText(name="Braised Pork Belly", description="Rich but not greasy",
                           ingredient_names={"pork_belly": "Pork Belly", "soy_sauce": "Light Soy Sauce"},
                           step_texts=["Sear the pork belly", "Braise on low for 40 min"],
                           display_tags=["Sichuan", "Home-style", "Savory"]),
        },
    )
    print("✅ 正例校验通过：", sample.recipe_id, "| i18n:", list(sample.i18n),
          "| unknown_facets:", sample.unknown_facets())

    # 3) 负例自测
    print("— 负例 —")
    try:
        Facets(spiciness=9)
        print("  ✗ spiciness=9 未被拒")
    except Exception as e:
        print("  ✓ spiciness=9 被拒")
    try:
        Recipe(recipe_id="x", canonical_lang="zh", i18n={"en": I18nText(name="x")})
        print("  ✗ canonical_lang 缺 i18n 未被拒")
    except Exception:
        print("  ✓ canonical_lang 缺 i18n 被拒")
    bad = Recipe(recipe_id="y", canonical_lang="zh",
                 facets=Facets(cuisine=["martian"]),
                 i18n={"zh": I18nText(name="y")})
    print("  ✓ 越表 slug 软校验返回:", bad.unknown_facets())
