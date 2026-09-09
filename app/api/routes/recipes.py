"""
菜谱相关 HTTP API 路由

提供菜谱详情的查询接口，从 PostgreSQL 读取。
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.recipe_detail_store import load_recipe_detail
from app.orchestrator.recipe_detail import format_recipe_detail

router = APIRouter()


class RecipeDetailResponse(BaseModel):
    """菜谱详情响应"""
    recipe_id: str
    language: str
    name: str
    introduction: Optional[str] = None
    tips: Optional[str] = None
    media: dict
    tags: list[str]
    ingredients: list[dict]
    steps: list[dict]
    cooking_time_seconds: Optional[int] = None
    servings: Optional[int] = None
    challenge_level: Optional[int] = None
    calorie_number: Optional[float] = None
    source_created_at: Optional[str] = None
    source_updated_at: Optional[str] = None
    formatted_markdown: str


class RecipeDetailNotFoundError(HTTPException):
    """菜谱详情未找到异常"""
    def __init__(self, recipe_id: str, language: str):
        super().__init__(
            status_code=404,
            detail=f"Recipe detail not found: recipe_id={recipe_id}, language={language}"
        )


@router.get("/{recipe_id}", response_model=RecipeDetailResponse)
async def get_recipe_detail(
    recipe_id: str,
    language: str = Query(default="zh", regex="^(zh|en)$"),
    include_markdown: bool = Query(default=True, description="是否包含格式化的 Markdown")
):
    """
    获取菜谱详情

    从 PostgreSQL 查询指定菜谱的详细信息，包括食材、步骤、烹饪时间等。

    - **recipe_id**: 菜谱 ID
    - **language**: 语言（zh 或 en）
    - **include_markdown**: 是否包含格式化的 Markdown（默认 true）

    返回菜谱的完整信息，包括：
    - 基本信息（名称、简介、技巧）
    - 媒体信息（图片、视频）
    - 食材列表（主料、调料）
    - 烹饪步骤
    - 烹饪时间、份数、难度等
    - 格式化的 Markdown（可选）
    """
    # 从 PostgreSQL 查询菜谱详情
    detail = await load_recipe_detail(recipe_id, language)

    if detail is None:
        raise RecipeDetailNotFoundError(recipe_id, language)

    # 构建响应
    response_data = {
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
    }

    # 如果需要格式化的 Markdown
    if include_markdown:
        formatted = format_recipe_detail(detail, language)
        response_data["formatted_markdown"] = formatted
    else:
        response_data["formatted_markdown"] = ""

    return response_data


@router.get("/{recipe_id}/markdown")
async def get_recipe_detail_markdown(
    recipe_id: str,
    language: str = Query(default="zh", regex="^(zh|en)$")
):
    """
    获取菜谱详情的 Markdown 格式

    只返回格式化的 Markdown 文本，适合直接展示。

    - **recipe_id**: 菜谱 ID
    - **language**: 语言（zh 或 en）
    """
    detail = await load_recipe_detail(recipe_id, language)

    if detail is None:
        raise RecipeDetailNotFoundError(recipe_id, language)

    formatted = format_recipe_detail(detail, language)

    return {
        "recipe_id": recipe_id,
        "language": language,
        "markdown": formatted
    }
