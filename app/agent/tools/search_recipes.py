"""搜索菜谱工具

封装现有的 RAG 检索逻辑，提供统一的工具接口。
"""

from typing import List, Optional
from app.agent.tools import Tool, ToolResult
from app.agent.recipe_search_service import RecipeSearchService


class SearchRecipesTool(Tool):
    """搜索菜谱工具

    使用 RAG 检索相关菜谱，返回匹配的菜谱列表。
    """

    name = "search_recipes"
    description = "搜索菜谱。根据用户描述的菜名、食材、口味等条件查找相关菜谱。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询，可以是菜名、食材、口味偏好等"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["query"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, query: str, top_k: int = 3) -> ToolResult:
        """执行菜谱搜索

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            # 调用 RAG 检索服务
            results = await self.search_service.search(
                query=query,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到与 '{query}' 相关的菜谱"
                )

            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "name": result.get("name"),
                    "score": result.get("score"),
                    "ingredients": result.get("ingredients", []),
                    "tags": result.get("tags", []),
                    "description": result.get("description", "")
                })

            return ToolResult.success(
                data={
                    "query": query,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个相关菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByIngredientsTool(Tool):
    """按食材搜索工具

    根据用户提供的食材列表搜索相关菜谱。
    """

    name = "search_by_ingredients"
    description = "根据食材搜索菜谱。当用户提到具体食材时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "ingredients": {
                "type": "array",
                "items": {"type": "string"},
                "description": "食材列表，例如 ['番茄', '鸡蛋']"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["ingredients"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, ingredients: List[str], top_k: int = 3) -> ToolResult:
        """执行按食材搜索

        Args:
            ingredients: 食材列表
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            # 构建搜索查询
            query = " ".join(ingredients)

            results = await self.search_service.search(
                query=query,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到包含 {', '.join(ingredients)} 的菜谱"
                )

            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "name": result.get("name"),
                    "score": result.get("score"),
                    "ingredients": result.get("ingredients", []),
                    "tags": result.get("tags", []),
                    "description": result.get("description", "")
                })

            return ToolResult.success(
                data={
                    "ingredients": ingredients,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个包含 {', '.join(ingredients)} 的菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByCuisineTool(Tool):
    """按菜系搜索工具

    根据用户指定的菜系搜索相关菜谱。
    """

    name = "search_by_cuisine"
    description = "根据菜系搜索菜谱。当用户提到具体菜系（如川菜、粤菜、西餐）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "cuisine": {
                "type": "string",
                "description": "菜系名称，例如 '川菜', '粤菜', '西餐'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["cuisine"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, cuisine: str, top_k: int = 3) -> ToolResult:
        """执行按菜系搜索

        Args:
            cuisine: 菜系名称
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            results = await self.search_service.search(
                query=cuisine,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到 {cuisine} 相关的菜谱"
                )

            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "name": result.get("name"),
                    "score": result.get("score"),
                    "ingredients": result.get("ingredients", []),
                    "tags": result.get("tags", []),
                    "description": result.get("description", "")
                })

            return ToolResult.success(
                data={
                    "cuisine": cuisine,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个 {cuisine} 菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )
