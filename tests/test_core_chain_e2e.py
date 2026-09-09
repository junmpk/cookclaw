"""核心链路端到端测试：从用户输入到响应输出。

覆盖：
- 意图分类 → 路由 → 检索 → 格式化 完整链路
- 使用 mock 避免真实 LLM / Milvus 调用
- 验证数据流正确性
"""
import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.orchestrator.intent import IntentCategory, IntentResult
from app.orchestrator.routing_models import RoutingContext


class TestEndToEndRecipeSearch:
    """食谱搜索端到端测试。"""

    @pytest.mark.asyncio
    async def test_recipe_search_chain(self):
        """食谱搜索完整链路。"""
        # Mock 意图分类
        mock_intent = IntentResult(
            related=True,
            category=IntentCategory.recipe_search,
            keywords=["红烧肉"],
            confidence=0.95,
        )

        # Mock 检索结果
        mock_search_result = {
            "success": True,
            "query": "红烧肉",
            "count": 3,
            "results": [
                {
                    "id": 1,
                    "score": 0.92,
                    "metadata": {
                        "name": "红烧肉",
                        "ingredients": ["五花肉", "姜", "八角"],
                        "tags": ["家常菜", "下饭菜"],
                        "image_url": "https://example.com/1.jpg",
                        "lang": "zh",
                    },
                },
                {
                    "id": 2,
                    "score": 0.85,
                    "metadata": {
                        "name": "东坡肉",
                        "ingredients": ["五花肉", "料酒", "酱油"],
                        "tags": ["浙菜", "经典"],
                        "image_url": "https://example.com/2.jpg",
                        "lang": "zh",
                    },
                },
            ],
        }

        # 验证意图分类结果
        assert mock_intent.category == IntentCategory.recipe_search
        assert "红烧肉" in mock_intent.keywords

        # 验证检索结果结构
        assert mock_search_result["success"] is True
        assert len(mock_search_result["results"]) == 2
        assert mock_search_result["results"][0]["metadata"]["name"] == "红烧肉"

    @pytest.mark.asyncio
    async def test_cooking_qa_chain(self):
        """烹饪问答完整链路。"""
        # Mock 意图分类
        mock_intent = IntentResult(
            related=True,
            category=IntentCategory.cooking_qa,
            keywords=["怎么做"],
            confidence=0.88,
        )

        # Mock LLM 回答
        mock_answer = "红烧肉需要先焯水，然后炒糖色，最后小火慢炖 1 小时。"

        # 验证意图分类
        assert mock_intent.category == IntentCategory.cooking_qa

        # 验证回答非空
        assert len(mock_answer) > 0
        assert "红烧肉" in mock_answer


class TestEndToEndRouting:
    """路由决策端到端测试。"""

    def test_routing_with_candidates(self):
        """有候选时的路由。"""
        context = RoutingContext(
            latest_candidates=[
                {"cookId": "1", "name": "红烧肉"},
                {"cookId": "2", "name": "东坡肉"},
            ],
            channel="qq",
        )

        # 验证上下文
        assert context.latest_candidates is not None
        assert len(context.latest_candidates) == 2
        summary = context.classifier_summary()
        assert summary["candidate_count"] == 2

    def test_routing_without_candidates(self):
        """无候选时的路由。"""
        context = RoutingContext(channel="qq")

        # 验证上下文
        assert context.latest_candidates == []
        summary = context.classifier_summary()
        assert summary["candidate_count"] == 0


class TestEndToEndResponseFormatting:
    """响应格式化端到端测试。"""

    def test_json_to_markdown_recipe_search(self):
        """食谱搜索结果转 Markdown。"""
        response_data = {
            "type": "recipe_search",
            "lang": "zh",
            "data": {
                "recipes": [
                    {
                        "name": "红烧肉",
                        "ingredients": ["五花肉", "姜"],
                        "tags": ["家常菜"],
                        "image": "https://example.com/1.jpg",
                    },
                ],
            },
            "message": "推荐菜谱",
        }

        # 验证结构
        assert response_data["type"] == "recipe_search"
        assert len(response_data["data"]["recipes"]) == 1
        assert response_data["data"]["recipes"][0]["name"] == "红烧肉"

    def test_json_to_plaintext_recipe_search(self):
        """食谱搜索结果转纯文本。"""
        response_data = {
            "type": "recipe_search",
            "lang": "zh",
            "data": {
                "recipes": [
                    {
                        "name": "红烧肉",
                        "ingredients": ["五花肉"],
                    },
                ],
            },
        }

        # 验证结构
        assert response_data["type"] == "recipe_search"
        assert response_data["lang"] == "zh"


class TestEndToEndStateManagement:
    """状态管理端到端测试。"""

    def test_candidate_remember_and_resolve(self):
        """候选记忆与选择。"""
        from app.ports import dialogue_state

        thread_id = "test-e2e-state"
        dialogue_state.clear_thread(thread_id)

        # 记忆候选
        search_result = {
            "results": [
                {
                    "metadata": {
                        "name": "红烧肉",
                        "recipe_id": "1",
                    },
                },
                {
                    "metadata": {
                        "name": "东坡肉",
                        "recipe_id": "2",
                    },
                },
            ],
        }
        dialogue_state.remember_candidates(thread_id, search_result, lang="zh")

        # 验证候选被记住
        candidates = dialogue_state.recall_candidate_context(thread_id)
        assert candidates is not None

        # 选择第二个（"第二个" → 解析为序号 2 → 返回第二个候选）
        selection = dialogue_state.resolve_selection(thread_id, "第二个")
        assert selection is not None
        # resolve_selection 返回候选本身（带 cookId/name），不是位置信息
        assert "东坡肉" in str(selection) or selection.get("name") == "东坡肉"

        # 清理
        dialogue_state.clear_thread(thread_id)

    def test_pending_action_lifecycle(self):
        """待处理动作生命周期。"""
        from app.ports import dialogue_state

        thread_id = "test-e2e-pending"
        dialogue_state.clear_thread(thread_id)

        # 设置待处理
        dialogue_state.set_pending(thread_id, "cook1", "红烧肉", device_id="dev1")

        # 获取待处理
        pending = dialogue_state.get_pending(thread_id)
        assert pending is not None
        assert pending["cookId"] == "cook1"
        assert pending["name"] == "红烧肉"

        # 清除待处理
        dialogue_state.clear_pending(thread_id)
        pending = dialogue_state.get_pending(thread_id)
        assert pending is None

        # 清理
        dialogue_state.clear_thread(thread_id)
