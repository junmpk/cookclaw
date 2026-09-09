"""高级搜索工具

提供更细粒度的搜索维度，让 LLM 能够根据用户的具体需求选择合适的工具。
"""

from typing import List, Optional
from app.agent.tools import Tool, ToolResult
from app.agent.recipe_search_service import RecipeSearchService


class SearchByCookingMethodTool(Tool):
    """按烹饪方法搜索工具

    根据用户指定的烹饪方法搜索菜谱。
    """

    name = "search_by_cooking_method"
    description = "根据烹饪方法搜索菜谱。当用户提到具体的烹饪方式（如炒、炖、蒸、烤、煮、煎等）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "烹饪方法，例如 '炒', '炖', '蒸', '烤', '煮', '煎', '炸', '凉拌'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["method"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, method: str, top_k: int = 3) -> ToolResult:
        """执行按烹饪方法搜索

        Args:
            method: 烹饪方法
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            results = await self.search_service.search(
                query=method,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到 {method} 相关的菜谱"
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
                    "method": method,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个 {method} 菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByTasteTool(Tool):
    """按口味搜索工具

    根据用户指定的口味偏好搜索菜谱。
    """

    name = "search_by_taste"
    description = "根据口味搜索菜谱。当用户提到口味偏好（如甜、咸、辣、酸、清淡、重口味等）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "taste": {
                "type": "string",
                "description": "口味偏好，例如 '甜', '咸', '辣', '酸', '清淡', '重口味', '麻辣', '酸甜'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["taste"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, taste: str, top_k: int = 3) -> ToolResult:
        """执行按口味搜索

        Args:
            taste: 口味偏好
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            results = await self.search_service.search(
                query=taste,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到 {taste} 口味的菜谱"
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
                    "taste": taste,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个 {taste} 口味的菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByDietaryTool(Tool):
    """按饮食限制搜索工具

    根据用户的饮食限制或特殊需求搜索菜谱。
    """

    name = "search_by_dietary"
    description = "根据饮食限制搜索菜谱。当用户提到饮食限制或特殊需求（如素食、低脂、低糖、无麸质、减肥餐等）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "dietary": {
                "type": "string",
                "description": "饮食限制，例如 '素食', '低脂', '低糖', '无麸质', '减肥餐', '增肌餐', '糖尿病饮食'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["dietary"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, dietary: str, top_k: int = 3) -> ToolResult:
        """执行按饮食限制搜索

        Args:
            dietary: 饮食限制
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            results = await self.search_service.search(
                query=dietary,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到符合 {dietary} 要求的菜谱"
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
                    "dietary": dietary,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个符合 {dietary} 要求的菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByCookingTimeTool(Tool):
    """按烹饪时间搜索工具

    根据用户期望的烹饪时间搜索菜谱。
    """

    name = "search_by_cooking_time"
    description = "根据烹饪时间搜索菜谱。当用户提到时间要求（如快手菜、15分钟内、30分钟内、1小时等）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "max_minutes": {
                "type": "integer",
                "description": "最大烹饪时间（分钟），例如 15, 30, 60"
            },
            "time_description": {
                "type": "string",
                "description": "时间描述，例如 '快手菜', '30分钟内', '1小时'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["max_minutes"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(
        self,
        max_minutes: int,
        time_description: Optional[str] = None,
        top_k: int = 3
    ) -> ToolResult:
        """执行按烹饪时间搜索

        Args:
            max_minutes: 最大烹饪时间（分钟）
            time_description: 时间描述（可选）
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            # 构建搜索查询
            query = f"{max_minutes}分钟" if not time_description else time_description

            results = await self.search_service.search(
                query=query,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到 {max_minutes} 分钟内能完成的菜谱"
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
                    "description": result.get("description", ""),
                    "cooking_time": result.get("cooking_time")
                })

            return ToolResult.success(
                data={
                    "max_minutes": max_minutes,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个 {max_minutes} 分钟内能完成的菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByDifficultyTool(Tool):
    """按难度搜索工具

    根据用户期望的难度级别搜索菜谱。
    """

    name = "search_by_difficulty"
    description = "根据难度搜索菜谱。当用户提到难度要求（如简单、中等、困难、新手友好等）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "difficulty": {
                "type": "string",
                "enum": ["简单", "中等", "困难", "新手友好"],
                "description": "难度级别"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["difficulty"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, difficulty: str, top_k: int = 3) -> ToolResult:
        """执行按难度搜索

        Args:
            difficulty: 难度级别
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            results = await self.search_service.search(
                query=difficulty,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到 {difficulty} 难度的菜谱"
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
                    "description": result.get("description", ""),
                    "difficulty": result.get("difficulty")
                })

            return ToolResult.success(
                data={
                    "difficulty": difficulty,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个 {difficulty} 难度的菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )


class SearchByOccasionTool(Tool):
    """按场合搜索工具

    根据用户的用餐场合搜索菜谱。
    """

    name = "search_by_occasion"
    description = "根据用餐场合搜索菜谱。当用户提到特定的用餐场合（如早餐、午餐、晚餐、聚会、野餐、约会等）时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "occasion": {
                "type": "string",
                "description": "用餐场合，例如 '早餐', '午餐', '晚餐', '聚会', '野餐', '约会', '工作餐'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3",
                "default": 3
            }
        },
        "required": ["occasion"]
    }

    def __init__(self, search_service: RecipeSearchService):
        self.search_service = search_service

    async def execute(self, occasion: str, top_k: int = 3) -> ToolResult:
        """执行按场合搜索

        Args:
            occasion: 用餐场合
            top_k: 返回结果数量

        Returns:
            ToolResult: 搜索结果
        """
        try:
            results = await self.search_service.search(
                query=occasion,
                top_k=top_k,
                include_metadata=True
            )

            if not results:
                return ToolResult.error(
                    message=f"未找到适合 {occasion} 的菜谱"
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
                    "occasion": occasion,
                    "count": len(formatted_results),
                    "results": formatted_results
                },
                message=f"找到 {len(formatted_results)} 个适合 {occasion} 的菜谱"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"搜索失败: {str(e)}"
            )
