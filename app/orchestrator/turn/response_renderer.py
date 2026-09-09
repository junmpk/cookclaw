"""领域响应适配与多通道 Renderer。"""

from __future__ import annotations

import json
from typing import Any

from app.orchestrator.turn.runtime_models import ResponseEnvelope


_RESERVED_RESPONSE_FIELDS = {"type", "intent", "lang", "data", "message"}


# ─── 自然 CTA 轮换 ──────────────────────────────────────────────────────
#
# 单菜回复可使用自然 CTA；多菜回复只在整组末尾保留一个入口，避免每张卡片
# 都催用户操作。用 index 做确定性选择，便于回归，不引入随机状态。

_DETAIL_CTA_ZH = (
    "想看完整做法？回「详情 {index}」就行 🍳",
    "想细看【{name}】的步骤和用量？回「详情 {index}」我铺开给你",
    "「详情 {index}」有完整食材和步骤，想试就点开",
    "对这道感兴趣的话，回「详情 {index}」看完整做法",
    "「详情 {index}」里有完整步骤，想做就直接打开",
    "【{name}】的完整做法在「详情 {index}」里",
)

_DETAIL_CTA_EN = (
    "Want the full recipe? Reply \"details {index}\" 🍳",
    "Curious about {name}? Reply \"details {index}\" for the full steps.",
    "Full ingredients & steps at \"details {index}\".",
    "Reply \"details {index}\" to see how it's made.",
    "Tap \"details {index}\" for the full walkthrough.",
    "The full recipe is at \"details {index}\".",
)


def _detail_cta(index: int, name: str = "", lang: str = "zh") -> str:
    """根据序号从自然话术里轮换，避免每条菜谱都用同一句客服腔 CTA。"""
    pool = _DETAIL_CTA_EN if lang == "en" else _DETAIL_CTA_ZH
    # 用序号做确定性选择：同一序号每次话术相同（便于测试/回归）
    template = pool[(index - 1) % len(pool)]
    return template.format(index=index, name=name or "")


def response_to_envelope(
    raw_response: str,
    *,
    handled_by: str = "conversation_fallback",
    trace_id: str | None = None,
) -> ResponseEnvelope:
    """在领域 adapter 边界只解析一次已有 JSON/string 响应。"""
    raw = str(raw_response or "")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        return ResponseEnvelope(
            response_type="text",
            message=raw,
            handled_by=handled_by,
            trace_id=trace_id,
        )
    data = payload.get("data")
    return ResponseEnvelope(
        response_type=str(payload.get("type") or "chat"),
        intent=(
            str(payload["intent"])
            if payload.get("intent") is not None
            else None
        ),
        lang=(str(payload["lang"]) if payload.get("lang") is not None else None),
        message=str(payload.get("message") or ""),
        data=dict(data) if isinstance(data, dict) else {},
        extra_fields={
            key: value
            for key, value in payload.items()
            if key not in _RESERVED_RESPONSE_FIELDS
        },
        channel_payload=payload,
        handled_by=handled_by,
        trace_id=trace_id,
    )


class IMResponseRenderer:
    """把 ResponseEnvelope 渲染回 QQ/微信/WhatsApp 共用的 JSON 契约。"""

    def render(self, envelope: ResponseEnvelope) -> str:
        payload = envelope.public_payload()
        if payload is None:
            return envelope.message
        return json.dumps(payload, ensure_ascii=False)


def _web_recipe_section(recipe: dict[str, Any], index: int, lang: str) -> list[str]:
    """把一条菜谱渲染成叙述式 Web Markdown 卡片。

    旧实现是僵硬的数据表格式：`### {idx}. {name}` + `**推荐理由**：...`。
    新实现更像朋友在推荐：菜名加粗 + 推荐理由直接引用 + 食材标签合并。
    """
    name = str(
        recipe.get("name")
        or ("Unknown recipe" if lang == "en" else "未知菜品")
    ).strip()
    image_url = str(recipe.get("image") or "").strip()
    ingredients = [
        str(item).strip()
        for item in list(recipe.get("ingredients") or [])[:8]
        if str(item).strip()
    ]
    tags = [
        str(item).strip()
        for item in list(recipe.get("tags") or [])[:6]
        if str(item).strip()
    ]
    reason = str(recipe.get("recommendation_reason") or "").strip()
    role = str(recipe.get("menu_role") or "").strip()

    # 菜名：不用 ### 僵硬标题，用数字 emoji + 加粗
    number_emoji = f"{index}️⃣" if index <= 10 else f"**{index}.**"
    lines = [f"{number_emoji} **{name}**", ""]
    if image_url.startswith(("http://", "https://")):
        safe_alt = name.replace("[", " ").replace("]", " ")
        lines.extend([f"![{safe_alt}]({image_url})", ""])
    if role:
        lines.append(f"_{role}_")
    # 推荐理由：直接作为引用块（无"推荐理由："标签）
    if reason:
        lines.extend([f"> {reason}", ""])
    # 食材 + 标签合并成一段（去掉字段标签，更像自然语言）
    details: list[str] = []
    if ingredients:
        value = "、".join(ingredients) if lang != "en" else ", ".join(ingredients)
        details.append(f"🥘 {value}")
    if tags:
        value = " · ".join(tags)
        details.append(f"🏷️ {value}")
    if details:
        lines.append("  ".join(details))
        lines.append("")
    return lines


def _collection_detail_cta(recipes: list[dict[str, Any]], lang: str) -> str:
    if len(recipes) == 1:
        name = str(recipes[0].get("name") or "").strip()
        return _detail_cta(1, name=name, lang=lang)
    return (
        "Want the full steps for one of these? Reply \"details\" plus its number."
        if lang == "en"
        else "想细看哪一道，回“详情 + 序号”就行。"
    )


class WebResponseRenderer:
    """把统一 Envelope 渲染为叙述式 Web Markdown。

    结构：自然过渡段落 → 菜谱卡片 1-N → 结尾引导。
    移除了旧的 "## 推荐菜谱" / "## 下一步" 僵硬数据表标题。
    """

    def render(self, envelope: ResponseEnvelope) -> str:
        if envelope.response_type not in {"recipe_search", "menu_plan"}:
            return str(envelope.message or "")
        lang = str(envelope.lang or "zh")
        data = dict(envelope.data or {})
        recipes = [
            dict(item)
            for item in list(data.get("recipes") or [])
            if isinstance(item, dict)
        ]
        if not recipes:
            return str(envelope.message or "")

        # 开场：直接用 envelope.message（LLM 生成的自然过渡段落）
        # 不再加 "## 推荐菜谱" 僵硬标题
        parts: list[str] = []
        opening = str(envelope.message or "").strip()
        if opening:
            parts.extend([opening, ""])
        # 菜谱卡片只陈列事实；整组最多保留一个主要 CTA。
        for index, recipe in enumerate(recipes, 1):
            parts.extend(_web_recipe_section(recipe, index, lang))
        # 结尾：直接放 closing，不再用 "## 下一步" 标题
        closing = str(data.get("closing") or "").strip()
        if closing:
            parts.append(closing)
        else:
            parts.append(_collection_detail_cta(recipes, lang))
        return "\n".join(parts).strip()


def prepend_response_ack(
    envelope: ResponseEnvelope,
    acknowledgement: str,
) -> ResponseEnvelope:
    """在结构化边界追加偏好确认，不重复解析通道 JSON。"""
    prefix = str(acknowledgement or "").strip()
    if not prefix:
        return envelope
    payload = envelope.public_payload()
    if payload is None:
        message = "\n\n".join(
            value
            for value in (prefix, str(envelope.message or "").strip())
            if value
        )
        return envelope.model_copy(update={"message": message})

    message = str(payload.get("message") or "").strip()
    payload["message"] = " ".join(
        value for value in (prefix, message) if value
    )
    data = payload.get("data")
    if isinstance(data, dict) and "opening" in data:
        opening = str(data.get("opening") or "").strip()
        data["opening"] = " ".join(
            value for value in (prefix, opening) if value
        )
    return envelope.model_copy(
        update={
            "message": str(payload.get("message") or ""),
            "data": dict(data) if isinstance(data, dict) else envelope.data,
            "channel_payload": payload,
        }
    )


def public_response_shape(envelope: ResponseEnvelope) -> dict[str, Any]:
    """供回放测试比较公开语义形状，不包含内部 trace 和 patch。"""
    payload = envelope.public_payload()
    return (
        payload
        if payload is not None
        else {"type": "text", "message": envelope.message}
    )
