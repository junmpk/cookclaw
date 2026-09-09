"""获取菜谱详情工具

从 PostgreSQL 查询菜谱详情，而非 Milvus 向量数据库。
"""

from typing import Optional
from app.agent.tools import Tool, ToolResult
from app.recipe_detail_store import load_recipe_detail
from app.orchestrator.recipe_detail import format_recipe_detail


class GetRecipeDetailsTool(Tool):
    """获取菜谱详情工具

    根据菜谱 ID 从 PostgreSQL 获取完整的菜谱信息。
    """

    name = "get_recipe_details"
    description = "获取指定菜谱的详细信息，包括食材清单、烹饪步骤、所需时间等。当用户想看某个菜谱的完整详情时使用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "recipe_id": {
                "type": "string",
                "description": "菜谱 ID"
            },
            "language": {
                "type": "string",
                "enum": ["zh", "en"],
                "description": "语言，默认 zh",
                "default": "zh"
            }
        },
        "required": ["recipe_id"]
    }

    async def execute(self, recipe_id: str, language: str = "zh") -> ToolResult:
        """获取菜谱详情

        Args:
            recipe_id: 菜谱 ID
            language: 语言（zh 或 en）

        Returns:
            ToolResult: 菜谱详情
        """
        try:
            # 从 PostgreSQL 查询菜谱详情
            detail = await load_recipe_detail(recipe_id, language)

            if detail is None:
                return ToolResult.error(
                    message=f"未找到菜谱 ID: {recipe_id}"
                )

            # 返回结构化的菜谱详情
            return ToolResult.success(
                data={
                    "recipe_id": detail.get("recipe_id", recipe_id),
                    "language": detail.get("language", language),
                    "name": detail.get("name", ""),
                    "introduction": detail.get("introduction"),
                    "tips": detail.get("tips"),
                    "media": detail.get("media", {}),
                    "tags": detail.get("tags", []),
                    "ingredients": detail.get("ingredients", []),
                    "steps": detail.get("steps", []),
                    "cooking_time_seconds": detail.get("cooking_time_seconds"),
                    "servings": detail.get("servings"),
                    "challenge_level": detail.get("challenge_level"),
                    "calorie_number": detail.get("calorie_number"),
                    "source_created_at": detail.get("source_created_at"),
                    "source_updated_at": detail.get("source_updated_at"),
                },
                message=f"菜谱详情: {detail.get('name', recipe_id)}"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"获取菜谱详情失败: {str(e)}"
            )


class GetCookingStepsTool(Tool):
    """获取烹饪步骤工具

    获取指定菜谱的详细烹饪步骤。
    """

    name = "get_cooking_steps"
    description = "获取指定菜谱的详细烹饪步骤，包括每个步骤的具体操作和注意事项。"
    parameters = {
        "type": "object",
        "properties": {
            "recipe_id": {
                "type": "string",
                "description": "菜谱 ID"
            },
            "language": {
                "type": "string",
                "enum": ["zh", "en"],
                "description": "语言，默认 zh",
                "default": "zh"
            }
        },
        "required": ["recipe_id"]
    }

    async def execute(self, recipe_id: str, language: str = "zh") -> ToolResult:
        """获取烹饪步骤

        Args:
            recipe_id: 菜谱 ID
            language: 语言（zh 或 en）

        Returns:
            ToolResult: 烹饪步骤
        """
        try:
            detail = await load_recipe_detail(recipe_id, language)

            if detail is None:
                return ToolResult.error(
                    message=f"未找到菜谱 ID: {recipe_id}"
                )

            steps = detail.get("steps", [])

            if not steps:
                return ToolResult.error(
                    message=f"菜谱 {recipe_id} 没有烹饪步骤"
                )

            # 格式化步骤
            formatted_steps = []
            for i, step in enumerate(steps, 1):
                formatted_steps.append({
                    "step_number": step.get("number", i),
                    "type": step.get("type", "manual"),
                    "description": step.get("description", ""),
                    "duration_seconds": step.get("duration_seconds"),
                    "image_url": step.get("image_url"),
                    "video_url": step.get("video_url"),
                    "parameters": step.get("parameters", []),
                })

            return ToolResult.success(
                data={
                    "recipe_id": recipe_id,
                    "recipe_name": detail.get("name", ""),
                    "total_steps": len(formatted_steps),
                    "steps": formatted_steps
                },
                message=f"菜谱 {detail.get('name', recipe_id)} 共有 {len(formatted_steps)} 个步骤"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"获取烹饪步骤失败: {str(e)}"
            )


class GetIngredientListTool(Tool):
    """获取食材清单工具

    获取指定菜谱的完整食材清单，包括主料和调料。
    """

    name = "get_ingredient_list"
    description = "获取指定菜谱的完整食材清单，包括主料和调料的详细用量。"
    parameters = {
        "type": "object",
        "properties": {
            "recipe_id": {
                "type": "string",
                "description": "菜谱 ID"
            },
            "language": {
                "type": "string",
                "enum": ["zh", "en"],
                "description": "语言，默认 zh",
                "default": "zh"
            }
        },
        "required": ["recipe_id"]
    }

    async def execute(self, recipe_id: str, language: str = "zh") -> ToolResult:
        """获取食材清单

        Args:
            recipe_id: 菜谱 ID
            language: 语言（zh 或 en）

        Returns:
            ToolResult: 食材清单
        """
        try:
            detail = await load_recipe_detail(recipe_id, language)

            if detail is None:
                return ToolResult.error(
                    message=f"未找到菜谱 ID: {recipe_id}"
                )

            ingredients = detail.get("ingredients", [])

            # 分离主料和调料
            food_ingredients = []
            seasoning_ingredients = []

            for ing in ingredients:
                if not isinstance(ing, dict):
                    continue

                group = ing.get("group", "main")
                formatted_ing = {
                    "name": ing.get("name", ""),
                    "amount": ing.get("amount"),
                    "unit": ing.get("unit"),
                    "remark": ing.get("remark", ""),
                }

                if group == "seasoning":
                    seasoning_ingredients.append(formatted_ing)
                else:
                    food_ingredients.append(formatted_ing)

            return ToolResult.success(
                data={
                    "recipe_id": recipe_id,
                    "recipe_name": detail.get("name", ""),
                    "servings": detail.get("servings"),
                    "food_ingredients": food_ingredients,
                    "seasoning_ingredients": seasoning_ingredients,
                },
                message=f"菜谱 {detail.get('name', recipe_id)} 的食材清单"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"获取食材清单失败: {str(e)}"
            )


class GetRecipeMarkdownTool(Tool):
    """获取菜谱 Markdown 工具

    获取指定菜谱的格式化 Markdown 详情。
    """

    name = "get_recipe_markdown"
    description = "获取指定菜谱的格式化 Markdown 详情，适合直接展示给用户。"
    parameters = {
        "type": "object",
        "properties": {
            "recipe_id": {
                "type": "string",
                "description": "菜谱 ID"
            },
            "language": {
                "type": "string",
                "enum": ["zh", "en"],
                "description": "语言，默认 zh",
                "default": "zh"
            }
        },
        "required": ["recipe_id"]
    }

    async def execute(self, recipe_id: str, language: str = "zh") -> ToolResult:
        """获取菜谱 Markdown

        Args:
            recipe_id: 菜谱 ID
            language: 语言（zh 或 en）

        Returns:
            ToolResult: 格式化的 Markdown
        """
        try:
            detail = await load_recipe_detail(recipe_id, language)

            if detail is None:
                return ToolResult.error(
                    message=f"未找到菜谱 ID: {recipe_id}"
                )

            # 生成格式化的 Markdown
            formatted = format_recipe_detail(detail, language)

            return ToolResult.success(
                data={
                    "recipe_id": recipe_id,
                    "recipe_name": detail.get("name", ""),
                    "language": language,
                    "markdown": formatted,
                },
                message=f"菜谱 {detail.get('name', recipe_id)} 的 Markdown 详情"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"获取菜谱 Markdown 失败: {str(e)}"
            )
