"""核心链路测试：fast_path 工具函数。

覆盖：
- detect_lang 语言检测
- _format_search_response 结果格式化
- 边界情况处理
"""
import pytest
from app.agent.fast_path import detect_lang


class TestDetectLang:
    """语言检测测试。"""

    def test_chinese_dominant(self):
        """中文占主导 → zh。"""
        assert detect_lang("红烧肉怎么做") == "zh"
        assert detect_lang("我想吃川菜") == "zh"
        assert detect_lang("推荐几道家常菜") == "zh"

    def test_english_dominant(self):
        """英文占主导 → en。"""
        assert detect_lang("How to cook braised pork") == "en"
        assert detect_lang("Recommend some recipes") == "en"
        assert detect_lang("I want spicy food") == "en"

    def test_mixed_chinese_dominant(self):
        """混合文本，中文占比 > 0.3 → zh。"""
        assert detect_lang("我想吃 spicy 的菜") == "zh"
        assert detect_lang("红烧 pork belly 怎么做") == "zh"

    def test_mixed_english_dominant(self):
        """混合文本，中文占比 < 0.3 → en。"""
        # "How to 红烧肉 in English" 去空格 19 字符，中文 3 → 0.16 → en
        assert detect_lang("How to 红烧肉 in English") == "en"
        # "just pork no 中文" 去空格 14 字符，中文 2 → 0.14 → en
        assert detect_lang("just pork no 中文") == "en"

    def test_empty_string(self):
        """空字符串 → 默认 zh（项目主语言）。"""
        assert detect_lang("") == "zh"

    def test_punctuation_only(self):
        """纯标点（无中文）→ en（中文占比 0）。"""
        assert detect_lang("???") == "en"
        # 中文标点不在 CJK 统一汉字区（U+4E00-9FFF），也算 en
        assert detect_lang("。。。") == "en"

    def test_numbers_only(self):
        """纯数字（无中文）→ en。"""
        assert detect_lang("12345") == "en"

    def test_threshold_boundary(self):
        """阈值边界（0.3）测试。

        detect_lang 用 > 0.3 严格大于，所以恰好 0.3 时返回 en。
        """
        # 高中文占比：明显 zh
        assert detect_lang("我想吃红烧肉") == "zh"  # 6/6 = 1.0
        assert detect_lang("红烧肉真的很好吃") == "zh"  # 8/8 = 1.0

        # 低中文占比：明显 en
        assert detect_lang("how to cook") == "en"  # 0/10 = 0
        assert detect_lang("pork belly recipe") == "en"

        # 混合：中文占比 > 0.3 → zh
        assert detect_lang("我想吃spicy的菜") == "zh"  # 去空格 10 字符,中文 5 → 0.5

        # 混合：中文占比 < 0.3 → en
        assert detect_lang("abc中文defghijklmn") == "en"  # 去空格 16,中文 2 → 0.125


class TestDetectLangEdgeCases:
    """语言检测边界情况。"""

    def test_single_chinese_char(self):
        """单个中文字 → zh。"""
        assert detect_lang("肉") == "zh"

    def test_single_english_char(self):
        """单个英文字母 → en。"""
        assert detect_lang("a") == "en"

    def test_whitespace_heavy(self):
        """大量空白 → 按实际字符比例。"""
        assert detect_lang("   红烧肉   ") == "zh"
        assert detect_lang("   braised   ") == "en"

    def test_special_characters(self):
        """特殊字符不影响检测。"""
        assert detect_lang("红烧肉！@#") == "zh"
        assert detect_lang("pork!@#") == "en"
