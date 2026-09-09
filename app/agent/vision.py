"""
视觉识别（多模态输入 -> 检索 query）。

用 Qwen-VL（DashScope，复用 DASHSCOPE_API_KEY）把图片识别成「成品菜」或
「冰箱/台面食材」，作为 RAG 检索的一种输入。

红线：VL 只产出 query 候选（菜名/食材），菜谱数据仍来自真实 recipe_search 检索。
VL 识别结果只用于召回，不自动执行设备。

职责边界：通道收图归通道层；本模块只做 图URL/base64 -> 结构化视觉识别结果，
作为语言无关的服务接口供上层调用。
"""
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

from app.observability.trace import observe_model_call, record_tool_call

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
_THRESHOLD = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.6"))

_client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    http_client=httpx.AsyncClient(trust_env=False),
)

_PROMPT = {
    "zh": """你是厨房图片识别器。请同时判断图片里是【成品菜】还是【冰箱/台面/购物袋里的生食材】。
可能有多张图片，要把所有图片合并理解。只输出 JSON，不要解释、不要代码块。

输出格式：
{
  "is_food": true/false,
  "scene_type": "dish|ingredients|mixed|not_food",
  "dish_name": "如果是成品菜，给最可能的中文菜名；如果主要是生食材则留空",
  "dish_names": ["可能的菜名，最多3个"],
  "ingredients": [
    {"name": "食材中文名", "confidence": 0.0, "state": "raw|cooked|packaged|unknown"}
  ],
  "search_query": "用于搜索菜谱的中文检索词",
  "confidence": 0.0
}

规则：
- 冰箱、案板、购物袋、包装盒里的肉菜蛋奶调料，按 ingredients 处理，不要硬猜成某道菜。
- 只写看得清或高度可能的可食用食材；不要把冰箱、碗盘、锅、包装品牌、人物、背景当食材。
- 食材去重并统一常用名，例如“番茄/西红柿”只保留一个；多图合计最多保留 12 个。
- 如果是生食材，dish_name 必须为空，search_query 用 3-5 个核心食材 + “家常菜”，例如“鸡蛋 西红柿 青椒 家常菜”。
- 如果是成品菜，search_query 优先用 dish_name；不确定时用主要食材组合。
- confidence 是整体把握，0~1。""",
    "en": """You are a kitchen image recognizer. Decide whether the image(s) show a prepared dish or raw fridge/pantry ingredients.
There may be multiple images; understand them together. Output JSON only, no explanation or code fences.

Schema:
{
  "is_food": true/false,
  "scene_type": "dish|ingredients|mixed|not_food",
  "dish_name": "most likely dish name if this is a prepared dish; empty for mostly raw ingredients",
  "dish_names": ["possible dish names, up to 3"],
  "ingredients": [
    {"name": "ingredient name", "confidence": 0.0, "state": "raw|cooked|packaged|unknown"}
  ],
  "search_query": "recipe-search query",
  "confidence": 0.0
}

Rules:
- is_food means the image contains any edible ingredient OR a prepared dish. Raw ingredients must set is_food=true.
- Fridge shelves, cutting boards, grocery bags, packages, raw meat/vegetables/eggs/dairy/seasonings are ingredients, not a dish.
- Include only visible or highly likely edible ingredients. Do not include fridge, cookware, dishes, package brands, people, or background.
- Deduplicate ingredients and use common names. Keep at most 12 across all images.
- For raw ingredients, dish_name must be empty and search_query should use 3-5 core ingredients plus 'recipe'.
- For prepared dishes, search_query should prefer dish_name; if uncertain, use the main ingredients.
- confidence is overall confidence, 0~1.""",
}


def _strip_json_fence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return raw


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(out, 1.0))


def _dedupe_text(values: list[str], limit: int = 8) -> list[str]:
    out = []
    seen = set()
    for value in values:
        s = str(value or "").strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
        if len(out) >= limit:
            break
    return out


def _normalize_ingredients(raw: Any, limit: int = 12) -> list[dict]:
    items = []
    seen = set()
    if not isinstance(raw, list):
        return []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("ingredient") or "").strip()
            confidence = _safe_float(item.get("confidence"), 0.0)
            state = str(item.get("state") or "unknown").strip() or "unknown"
        else:
            name = str(item or "").strip()
            confidence = 0.0
            state = "unknown"
        if not name or name in seen:
            continue
        items.append({"name": name, "confidence": confidence, "state": state})
        seen.add(name)
        if len(items) >= limit:
            break
    return items


def _image_url_from_input(image: dict) -> str:
    image_url = image.get("image_url")
    if image_url:
        return str(image_url)
    image_base64 = image.get("image_base64")
    if not image_base64:
        return ""
    image_mime = image.get("image_mime") or "image/jpeg"
    return f"data:{image_mime};base64,{image_base64}"


def _empty_recognition(error: str = "") -> dict:
    out = {
        "is_food": False,
        "scene_type": "not_food",
        "dish_name": "",
        "dish_names": [],
        "ingredients": [],
        "search_query": "",
        "confidence": 0.0,
        "is_dish": False,
    }
    if error:
        out["error"] = error
    return out


def _normalize_recognition(obj: dict, lang: str = "zh") -> dict:
    scene_type = str(obj.get("scene_type") or "").strip().lower()
    if scene_type not in {"dish", "ingredients", "mixed", "not_food"}:
        if obj.get("is_food") is False:
            scene_type = "not_food"
        elif obj.get("dish_name"):
            scene_type = "dish"
        elif obj.get("ingredients"):
            scene_type = "ingredients"
        else:
            scene_type = "not_food"

    dish_name = str(obj.get("dish_name") or "").strip()
    ingredients = _normalize_ingredients(obj.get("ingredients"), limit=12)
    dish_names = _dedupe_text([dish_name] + list(obj.get("dish_names") or []), limit=3)
    search_query = str(obj.get("search_query") or "").strip()

    if scene_type == "ingredients":
        dish_name = ""
    # 模型偶尔会把 is_food 理解成“是否为成品菜”，从而返回
    # scene_type=ingredients + 非空食材 + is_food=false。以结构化食材/菜名证据
    # 校正这个自相矛盾的布尔值，避免把已识别成功的冰箱照片当成非食物。
    if ingredients and scene_type == "not_food":
        scene_type = "ingredients"
        dish_name = ""
    elif dish_name and scene_type == "not_food":
        scene_type = "dish"
    if not search_query:
        if scene_type == "dish" and dish_name:
            search_query = dish_name
        elif ingredients:
            names = [item["name"] for item in ingredients[:5]]
            suffix = "家常菜" if lang != "en" else "recipe"
            search_query = " ".join([*names, suffix])

    has_food_evidence = bool(ingredients) or bool(dish_name)
    is_food = has_food_evidence or bool(
        obj.get("is_food", scene_type != "not_food")
    )
    if scene_type == "not_food" and not has_food_evidence:
        is_food = False

    return {
        "is_food": is_food,
        "scene_type": scene_type,
        "dish_name": dish_name,
        "dish_names": dish_names,
        "ingredients": ingredients,
        "search_query": search_query,
        "confidence": _safe_float(obj.get("confidence"), 0.0),
        # 兼容旧调用方：is_dish 只表示“成品菜”，不再把冰箱食材误当菜。
        "is_dish": scene_type in {"dish", "mixed"} and bool(dish_name),
    }


async def recognize_food_images(images: list[dict], lang: str = "zh", timeout: int = 45,
                                max_images: int = 6) -> dict:
    """Qwen-VL 识别一组厨房图片 -> 成品菜/食材结构化结果。"""
    started = asyncio.get_running_loop().time()
    usable = []
    for image in images or []:
        url = _image_url_from_input(image)
        if url:
            usable.append((image, url))
        if len(usable) >= max_images:
            break
    if not usable:
        record_tool_call(
            "vision_recognition",
            duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            success=False,
            error_type="no_image",
        )
        return _empty_recognition("no image")

    prompt = _PROMPT.get(lang, _PROMPT["zh"])
    content = [{"type": "text", "text": prompt}]
    for idx, (_, url) in enumerate(usable, 1):
        content.append({"type": "text", "text": f"图片 {idx} / image {idx}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})

    try:
        resp = await observe_model_call(
            lambda: _client.chat.completions.create(
                model=_VL_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=600,
                timeout=timeout,
            ),
            stage="vision_model",
            model=_VL_MODEL,
        )
        raw = _strip_json_fence(resp.choices[0].message.content or "")
        obj = json.loads(raw)
        result = _normalize_recognition(obj, lang=lang)
        result["image_count"] = len(usable)
        record_tool_call(
            "vision_recognition",
            duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            success=True,
        )
        return result
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        logger.warning(
            "[图片识别] 视觉模型调用失败：model=%s error_type=%s status=%s",
            _VL_MODEL,
            type(e).__name__,
            status_code if status_code is not None else "-",
        )
        # 上层只需区分“服务失败”和“模型判定非食物”；不要把第三方原始报错、
        # 请求体或签名 URL 带进响应和日志。
        error_type = type(e).__name__
        record_tool_call(
            "vision_recognition",
            duration_ms=(asyncio.get_running_loop().time() - started) * 1000,
            success=False,
            timeout="timeout" in error_type.lower(),
            error_type=error_type,
        )
        return _empty_recognition(type(e).__name__)


async def recognize_dish(image_url: Optional[str] = None, image_base64: Optional[str] = None,
                         lang: str = "zh", timeout: int = 30, image_mime: str = "image/jpeg") -> dict:
    """兼容旧接口：单图识别，返回旧字段并附带新字段。"""
    image = {"image_url": image_url} if image_url else {
        "image_base64": image_base64,
        "image_mime": image_mime,
    }
    rec = await recognize_food_images([image], lang=lang, timeout=timeout, max_images=1)
    legacy_ingredients = [item["name"] for item in rec.get("ingredients") or []]
    out = dict(rec)
    out["ingredients"] = legacy_ingredients
    return out


async def search_by_image(image_url: Optional[str] = None, image_base64: Optional[str] = None,
                          lang: str = "zh", top_k: int = 3) -> dict:
    """图 -> VLM 识别 -> recipe_search（真实检索）。"""
    image = {"image_url": image_url} if image_url else {"image_base64": image_base64}
    rec = await recognize_food_images([image], lang=lang)
    query = rec.get("search_query") or rec.get("dish_name")
    if not rec.get("is_food") or not query or rec.get("confidence", 0.0) < _THRESHOLD:
        return {"recognized": rec, "need_confirm": True, "search": None}
    from app.agent.recipe_search_service import search as _search
    result = await _search(query, top_k=top_k, lang=lang)
    return {"recognized": rec, "need_confirm": False, "search": result}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python -m app.agent.vision <image_url> [lang]")
        sys.exit(1)
    url, lang = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "zh")
    out = asyncio.run(recognize_dish(image_url=url, lang=lang))
    print(json.dumps(out, ensure_ascii=False, indent=2))
