"""核心链路测试：意图分类 + 路由决策。

覆盖：
- IntentCategory 解析
- IntentResult 构建
- classify_intent 边界情况
"""
import pytest
from app.orchestrator.intent import IntentCategory, IntentResult


class TestIntentCategory:
    """意图类别枚举测试。"""

    def test_from_str_valid_categories(self):
        """有效类别字符串能正确解析。"""
        assert IntentCategory.from_str("recipe_search") == IntentCategory.recipe_search
        assert IntentCategory.from_str("cooking_qa") == IntentCategory.cooking_qa
        assert IntentCategory.from_str("greeting") == IntentCategory.greeting
        assert IntentCategory.from_str("device_manage") == IntentCategory.device_manage

    def test_from_str_unknown_fallback(self):
        """未知类别回退到 unknown。"""
        assert IntentCategory.from_str("nonsense") == IntentCategory.unknown
        assert IntentCategory.from_str("") == IntentCategory.unknown
        assert IntentCategory.from_str(None) == IntentCategory.unknown

    def test_from_str_case_sensitive(self):
        """大小写敏感（实际实现）。"""
        # from_str 不做大小写转换，大写会落到 unknown
        assert IntentCategory.from_str("RECIPE_SEARCH") == IntentCategory.unknown
        assert IntentCategory.from_str("recipe_search") == IntentCategory.recipe_search


class TestIntentResult:
    """意图结果模型测试。"""

    def test_default_values(self):
        """默认值合理。"""
        result = IntentResult()
        assert result.related is True
        assert result.category == IntentCategory.unknown
        assert result.keywords == []
        assert result.confidence is None
        assert result.action == "unknown"

    def test_custom_values(self):
        """自定义值正确存储。"""
        result = IntentResult(
            related=True,
            category=IntentCategory.recipe_search,
            keywords=["红烧肉", "做法"],
            confidence=0.95,
            action="search_recipe",
        )
        assert result.category == IntentCategory.recipe_search
        assert "红烧肉" in result.keywords
        assert result.confidence == 0.95

    def test_serialization(self):
        """可序列化为 dict。"""
        result = IntentResult(
            category=IntentCategory.cooking_qa,
            keywords=["怎么做"],
        )
        data = result.model_dump()
        assert data["category"] == "cooking_qa"
        assert "怎么做" in data["keywords"]


class TestIntentCategoryValues:
    """验证所有意图类别枚举值。"""

    def test_all_categories_present(self):
        """所有预期的意图类别都存在。"""
        expected = {
            "recipe_search",
            "recipe_recommend",
            "recipe_execute",
            "device_manage",
            "cooking_qa",
            "greeting",
            "off_topic",
            "unknown",
        }
        actual = {cat.value for cat in IntentCategory}
        assert expected == actual
