import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import uvicorn
from io import BytesIO
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.agent.participle_agent import qqbot_chat
from app.agent.deep_agent_rollout import deep_agent_rollout_decision
from app.api.routes.chat import router as chat_router
from app.api.routes.recipes import router as recipes_router
from app.core.config import settings
from app.observability.trace import (
    ensure_turn_trace,
    finish_turn_trace,
    trace_stage,
)
from app.ports.dialogue_state import normalize_voice_control_text

logger = logging.getLogger(__name__)


def _require_webhook_secret(
    expected: str,
    provided: str | None,
    *,
    channel: str,
) -> None:
    """只允许持有共享密钥的本机 sidecar 投递消息，不记录密钥内容。"""
    configured = str(expected or "")
    if not configured:
        logger.error("%s webhook 被拒绝：服务端未配置共享密钥", channel)
        raise HTTPException(
            status_code=503,
            detail="Webhook authentication is not configured",
        )
    if provided is None or not secrets.compare_digest(configured, str(provided)):
        logger.warning("%s webhook 被拒绝：共享密钥缺失或不匹配", channel)
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")


def _plaintext_message_logs_enabled() -> bool:
    return os.getenv("LOG_PLAINTEXT_MESSAGES", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _log_identifier(value: object) -> str:
    """本地调试可输出原值；生产默认只保留稳定短指纹。"""
    raw = str(value or "")
    if not raw:
        return "-"
    if _plaintext_message_logs_enabled():
        return raw
    return "sha256:" + hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _log_content(value: object) -> str:
    """本地调试可输出正文；生产默认只记长度。"""
    raw = str(value or "")
    if _plaintext_message_logs_enabled():
        return raw
    return f"[redacted chars={len(raw)}]"


def _log_url_origin(value: object) -> str:
    """日志仅保留 URL 协议/主机/端口，去掉签名路径与 query。"""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(str(value or ""))
        host = parsed.hostname or "-"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme or '-'}://{host}{port}"
    except (TypeError, ValueError):
        return "invalid-url"

_RECIPE_TEXT = {
    "zh": {
        "found_header": "可以先看这几个：",
        "unknown_recipe": "未知食谱",
        "unknown_dish": "未知菜品",
        "image": "图片",
        "ingredients": "食材",
        "tags": "标签",
        "tip": "💡 想做哪道，跟我说一声就行（比如「做第一道」）",
        "joiner": "、",
    },
    "en": {
        "found_header": "These are the closest fits:",
        "unknown_recipe": "Unknown recipe",
        "unknown_dish": "Unknown dish",
        "image": "Image",
        "ingredients": "Ingredients",
        "tags": "Tags",
        "tip": "💡 Tell me which one you'd like to cook (for example, \"make the first one\").",
        "joiner": ", ",
    },
}


def _response_lang(data: dict) -> str:
    """读取 Agent JSON 里的响应语言，未知时保持中文兼容旧数据。"""
    resp_data = data.get("data") if isinstance(data, dict) else {}
    lang = data.get("lang") or (resp_data or {}).get("lang")
    return "en" if lang == "en" else "zh"


def _menu_role_label(recipe: dict, lang: str) -> str:
    """菜单中的菜/汤角色提示；普通 recipe_search 不附加标签。"""
    role = str((recipe or {}).get("menu_role") or "").strip()
    if not role:
        return ""
    if lang == "en":
        return {
            "soup": "Soup",
            "scoped_dish": "Guest preference dish",
            "dish": "Dish",
        }.get(role, "Dish")
    return {
        "soup": "汤",
        "scoped_dish": "照顾局部偏好的菜",
        "dish": "菜",
    }.get(role, "菜")


def _recipe_detail_reference(recipe: dict, lang: str) -> tuple[str, bool]:
    """QQ 卡片的简短做饭参考；AI 详情只展示，不参与设备能力判断。"""
    detail = (recipe or {}).get("recipe_detail")
    if not isinstance(detail, dict):
        return "", False
    values = []
    try:
        seconds = int(float(detail.get("cooking_time_seconds") or 0))
    except (TypeError, ValueError):
        seconds = 0
    if seconds > 0:
        if seconds < 3600:
            minutes = max(1, round(seconds / 60))
            values.append(f"about {minutes} min" if lang == "en" else f"约{minutes}分钟")
        else:
            hours, remainder = divmod(seconds, 3600)
            minutes = round(remainder / 60)
            if lang == "en":
                values.append(f"about {hours} hr {minutes} min" if minutes else f"about {hours} hr")
            else:
                values.append(f"约{hours}小时{minutes}分钟" if minutes else f"约{hours}小时")
    servings = detail.get("servings")
    if isinstance(servings, (int, float)) and not isinstance(servings, bool) and servings > 0:
        value = f"{servings:g}" if isinstance(servings, float) else str(servings)
        values.append(f"{value} servings" if lang == "en" else f"{value}人份")
    steps = detail.get("steps")
    if isinstance(steps, list) and steps:
        values.append(f"{len(steps)} steps" if lang == "en" else f"{len(steps)}步")
    return " · ".join(values), bool(detail.get("ai_generated"))


def _recipe_text(lang: str, key: str) -> str:
    return _RECIPE_TEXT.get(lang, _RECIPE_TEXT["zh"])[key]


def _recipe_reasoning(data: dict) -> str:
    resp_data = data.get("data") if isinstance(data, dict) else {}
    reasoning = str((resp_data or {}).get("reasoning") or "").strip()
    if not reasoning:
        return ""
    try:
        max_chars = max(80, min(int(os.getenv("RECIPE_REASONING_MAX_CHARS", "220")), 500))
    except (TypeError, ValueError):
        max_chars = 220
    if len(reasoning) <= max_chars:
        return reasoning
    clipped = reasoning[:max_chars]
    for sep in ("。", "；", "，", ".", ";", ","):
        pos = clipped.rfind(sep)
        if pos >= max_chars * 0.65:
            return clipped[:pos + 1].rstrip()
    return clipped.rstrip() + "..."


def _natural_ingredient_sentence(ingredients, lang: str) -> str:
    """把真实食材写成一句人话；不把标签、相似度等数据库字段堆给用户。"""
    values = [str(item).strip() for item in (ingredients or []) if str(item).strip()]
    if not values:
        return ""
    joined = _recipe_text(lang, "joiner").join(values)
    return f"It uses {joined}." if lang == "en" else f"主要用到{joined}。"


def _has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def _has_latin(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in text)


def _expand_display_tag(tag: str) -> list[str]:
    tag = str(tag or "").strip()
    if not tag:
        return []
    pieces = [tag]
    if " / " in tag:
        pieces = [p.strip() for p in tag.split("/") if p.strip()]
    out: list[str] = []
    for piece in pieces:
        out.extend(p.strip() for p in re.split(r"[、,，;；\|]+", piece) if p.strip())
    return out


def _display_tags(tags, lang: str) -> list[str]:
    expanded: list[str] = []
    for raw in tags or []:
        expanded.extend(_expand_display_tag(raw))

    filtered: list[str] = []
    for tag in expanded:
        has_cjk = _has_cjk(tag)
        has_latin = _has_latin(tag)
        if lang == "en":
            if has_latin and not has_cjk:
                filtered.append(tag)
        elif has_cjk:
            filtered.append(tag)

    if not filtered:
        filtered = expanded

    out: list[str] = []
    seen = set()
    for tag in filtered:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
        if len(out) >= 4:
            break
    return out

# ─── 运行期全局状态 ──────────────────────────────────────────────────
# 这些对象由 lifespan() 统一创建和销毁。
# 放在模块级别，方便消息回调和路由直接访问同一份连接实例。

# QQ Bot 全局实例
_qq_adapter = None
_qq_task = None
_conversation_health_task = None
_im_graph_bridge = None

# WhatsApp 全局实例
_wa_adapter = None
_wa_subprocess = None

# 微信机器人全局实例
_wx_adapter = None
_wx_subprocess = None


# ─── 微信会话 token 持久化（扫码登录后保存，重启复用，过期再扫；不写死 .env）────
_WX_TOKEN_FILE = os.path.join(os.path.dirname(__file__), "weixinbot", ".wx_session.json")


def _load_wx_session() -> dict:
    """读取最近保存的微信会话，兼容旧的单账号调用。"""
    sessions = _load_wx_sessions()
    if not sessions:
        return {}
    return max(sessions.values(), key=lambda item: float(item.get("saved_at") or 0))


def _load_wx_sessions() -> dict[str, dict]:
    """读取全部微信账号；自动兼容并迁移旧版单账号 JSON 结构。"""
    try:
        with open(_WX_TOKEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return {}
        accounts = data.get("accounts")
        if isinstance(accounts, dict):
            return {
                str(account_id): dict(session)
                for account_id, session in accounts.items()
                if account_id and isinstance(session, dict) and session.get("token")
            }
        if data.get("token"):
            account_id = str(data.get("account_id") or "default")
            return {account_id: dict(data)}
        return {}
    except Exception:
        return {}


def _load_wx_token() -> str:
    """读取上次扫码登录持久化的微信会话 token（没有则空串）。"""
    return _load_wx_session().get("token", "") or ""


def _save_wx_session(
    token: str,
    *,
    base_url: str = "",
    account_id: str = "",
    user_id: str = "",
) -> None:
    """按 account_id 增量持久化扫码会话，不覆盖其他微信账号。"""
    import time
    try:
        resolved_account_id = account_id or "default"
        accounts = _load_wx_sessions()
        previous = accounts.get(resolved_account_id) or {}
        payload = {
            "token": token,
            "base_url": base_url or previous.get("base_url", ""),
            "account_id": resolved_account_id,
            "user_id": user_id or previous.get("user_id", ""),
            "saved_at": time.time(),
        }
        accounts[resolved_account_id] = payload
        temp_path = f"{_WX_TOKEN_FILE}.{os.getpid()}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "accounts": accounts}, f, ensure_ascii=False)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, _WX_TOKEN_FILE)
        logger.info("微信扫码会话已持久化：account=%s", _log_identifier(resolved_account_id))
    except Exception as e:
        logger.warning("微信扫码会话持久化失败：error_type=%s", type(e).__name__)


def _clear_wx_token(account_id: str = "") -> None:
    """删除一个过期账号；未指定账号时清理全部本地微信登录态。"""
    try:
        if not account_id:
            if os.path.exists(_WX_TOKEN_FILE):
                os.remove(_WX_TOKEN_FILE)
                logger.info("已清理全部过期微信会话 token")
            return
        accounts = _load_wx_sessions()
        accounts.pop(account_id, None)
        if not accounts:
            if os.path.exists(_WX_TOKEN_FILE):
                os.remove(_WX_TOKEN_FILE)
        else:
            temp_path = f"{_WX_TOKEN_FILE}.{os.getpid()}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"version": 2, "accounts": accounts}, f, ensure_ascii=False)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, _WX_TOKEN_FILE)
        logger.info("已清理过期微信会话 token：account=%s", _log_identifier(account_id))
    except Exception as e:
        logger.warning("清理过期微信会话 token 失败：error_type=%s", type(e).__name__)


async def _wx_restart_listening(account_id: str = "") -> None:
    """幂等启动指定账号；未指定时启动全部已注册账号。"""
    if _wx_adapter is None:
        return
    await _wx_adapter.start(account_id)


async def _handle_wx_login_result(result: dict) -> dict:
    """扫码成功后原子切换 token、账号和微信分配的 API 网关。"""
    token = ""
    if isinstance(result, dict):
        token = result.get("botToken") or result.get("token") or ""
    if token and _wx_adapter:
        base_url = result.get("baseUrl") or ""
        account_id = result.get("accountId") or ""
        user_id = result.get("userId") or ""
        _save_wx_session(
            token,
            base_url=base_url,
            account_id=account_id,
            user_id=user_id,
        )
        try:
            await _wx_adapter.set_token(token, base_url, account_id, user_id)
            logger.info("微信扫码登录成功，已切换 token、账号和 API 网关")
        except Exception as e:
            logger.warning("微信扫码登录后切换会话失败：error_type=%s", type(e).__name__)
    return result


def _json_to_markdown(data: dict) -> str:
    """将 QQ Bot Agent 返回的 JSON 数据组装成 Markdown 格式"""
    # 这里是“展示层”转换器：
    # Agent 负责产出结构化 JSON，这里负责把 JSON 变成 QQ/IM 友好的文本。
    resp_type = data.get("type", "")
    resp_data = data.get("data", {})
    message = data.get("message", "")

    # ─── 食谱搜索 ──────────────────────────────────────
    if resp_type in {"recipe_search", "menu_plan"} and resp_data.get("recipes"):
        lang = _response_lang(data)
        if resp_type == "menu_plan":
            title = "## 🍽️ Menu plan" if lang == "en" else "## 🍽️ 推荐菜单"
        else:
            title = "## 🍳 Recipe ideas" if lang == "en" else "## 🍳 推荐菜谱"
        parts = [title, ""]
        reasoning = _recipe_reasoning(data)
        opening = str((resp_data or {}).get("opening") or "").strip()
        strategy = str((resp_data or {}).get("strategy") or "").strip()
        if resp_type == "menu_plan" and not opening:
            opening = str(message or "").strip()
        if opening:
            parts.extend([opening, ""])
        elif reasoning:
            parts.extend([reasoning, ""])
        if not opening and not reasoning:
            parts.append(f"{_recipe_text(lang, 'found_header')}\n")
        for i, recipe in enumerate(resp_data["recipes"], 1):
            name = recipe.get("name", _recipe_text(lang, "unknown_recipe"))
            image = recipe.get("image", "")
            ingredients = recipe.get("ingredients", [])
            tags = recipe.get("tags", [])
            recommendation_reason = str(recipe.get("recommendation_reason") or "").strip()

            parts.extend([f"### {i}. {name}", ""])
            role_label = _menu_role_label(recipe, lang)
            if image:
                parts.extend([f"![{name} #260px #195px]({image})", ""])
            if role_label:
                parts.append(
                    f"- **🍽️ Menu role:** {role_label}"
                    if lang == "en" else f"- **🍽️ 菜单定位**：{role_label}"
                )
            if recommendation_reason:
                parts.extend([
                    (
                        f"> **Why it fits:** {recommendation_reason}"
                        if lang == "en" else f"> **推荐理由**：{recommendation_reason}"
                    ),
                    "",
                ])
            visible_ingredients = [
                str(item).strip() for item in ingredients if str(item).strip()
            ][:8]
            if visible_ingredients:
                ingredient_text = (
                    ", ".join(visible_ingredients)
                    if lang == "en" else "、".join(visible_ingredients)
                )
                parts.append(
                    f"- **🥘 Main ingredients:** {ingredient_text}"
                    if lang == "en" else f"- **🥘 主要食材**：{ingredient_text}"
                )
            visible_tags = [str(tag).strip() for tag in tags if str(tag).strip()][:6]
            if visible_tags:
                parts.append(
                    f"- **🏷️ Tags:** {' · '.join(visible_tags)}"
                    if lang == "en" else f"- **🏷️ 标签**：{' · '.join(visible_tags)}"
                )
            detail_reference, _ = _recipe_detail_reference(recipe, lang)
            if detail_reference:
                parts.append(
                    f"- **⏱️ Cooking reference:** {detail_reference}"
                    if lang == "en" else f"- **⏱️ 做饭参考**：{detail_reference}"
                )
            parts.extend([
                "",
                (
                    f'👉 Reply **"details {i}"** to open the full recipe.'
                    if lang == "en" else f"👉 回复 **「详情 {i}」** 查看完整菜谱。"
                ),
                "",
            ])
        closing = str((resp_data or {}).get("closing") or "").strip()
        if strategy or closing:
            parts.extend([
                "## How to choose" if lang == "en" else "## 怎么选",
                "",
            ])
            if strategy:
                parts.extend([strategy, ""])
            if closing:
                parts.append(closing)
        return "\n".join(parts)

    # ─── 公共联网搜索 ──────────────────────────────────
    if resp_type == "web_search":
        return message

    # ─── 设备状态 ──────────────────────────────────────
    if resp_type == "device_status":
        parts = [f"## 🔧 {message}\n"]
        if resp_data.get("device_online"):
            parts.append("- 设备状态: ✅ 在线")
        else:
            parts.append("- 设备状态: ❌ 离线")
        if resp_data.get("recipe_name"):
            parts.append(f"- 食谱: **{resp_data['recipe_name']}**")
        return "\n".join(parts)

    # ─── 烹饪进度 ──────────────────────────────────────
    if resp_type == "cooking_progress":
        progress = resp_data.get("progress", 0)
        step = resp_data.get("step", "?")
        total = resp_data.get("total_steps", "?")
        step_name = resp_data.get("step_name", "")
        remaining = resp_data.get("remaining_minutes", "?")
        recipe_name = resp_data.get("recipe_name", "")

        # 进度条
        filled = int(progress / 10)
        bar = "█" * filled + "░" * (10 - filled)

        parts = [
            f"## 👨‍🍳 {recipe_name} — 烹饪中\n",
            f"**进度**: [{bar}] {progress}%\n",
            f"- 当前步骤: 第{step}/{total}步 — **{step_name}**",
            f"- 预计剩余: **{remaining}分钟**\n",
            f"{message}",
        ]
        return "\n".join(parts)

    # ─── 烹饪完成 ──────────────────────────────────────
    if resp_type == "cooking_complete":
        recipe_name = resp_data.get("recipe_name", "")
        total_min = resp_data.get("total_minutes", "?")
        parts = [
            f"## 🎉 烹饪完成！\n",
            f"**{recipe_name}** 已完成，总用时 **{total_min}分钟**\n",
            f"{message}",
        ]
        return "\n".join(parts)

    # ─── 烹饪异常 ──────────────────────────────────────
    if resp_type == "cooking_error":
        error_code = resp_data.get("error_code", "")
        error_msg = resp_data.get("error_message", "")
        parts = [
            f"## ❌ 烹饪异常\n",
            f"**错误码**: `{error_code}`",
            f"**原因**: {error_msg}\n",
            f"{message}",
        ]
        return "\n".join(parts)

    # ─── 烹饪问答 ──────────────────────────────────────
    if resp_type == "cooking_qa":
        answer = resp_data.get("answer", message)
        parts = [
            f"## 💡 烹饪小贴士\n",
            answer,
        ]
        return "\n".join(parts)

    # ─── 问候 / 其他 ───────────────────────────────────
    if message:
        return message

    # 兜底：直接返回原文
    import json
    return json.dumps(data, ensure_ascii=False, indent=2)


def _qq_markdown_messages(data: dict) -> list[str]:
    """QQ 多图菜单分两条发送，规避单条 Markdown 第四张外链图偶发不渲染。"""
    rendered = _json_to_markdown(data)
    resp_data = data.get("data") if isinstance(data, dict) else {}
    recipes = (resp_data or {}).get("recipes") or []
    if data.get("type") != "menu_plan" or len(recipes) <= 3:
        return [rendered]

    fourth = recipes[3] if isinstance(recipes[3], dict) else {}
    fourth_name = str(
        fourth.get("name") or _recipe_text(_response_lang(data), "unknown_recipe")
    ).strip()
    marker = f"\n### 4. {fourth_name}\n"
    split_at = rendered.find(marker)
    if split_at < 0:
        logger.warning("[QQBot] 多图菜单未找到第 4 道分隔点，保持单条发送")
        return [rendered]

    first = rendered[:split_at].rstrip()
    remainder = rendered[split_at + 1:].lstrip()
    lang = _response_lang(data)
    role = str(fourth.get("menu_role") or "").strip()
    if role == "soup":
        continuation = "## 🍲 Soup" if lang == "en" else "## 🍲 汤"
    else:
        continuation = (
            "## 🍽️ Menu plan · continued"
            if lang == "en" else "## 🍽️ 推荐菜单 · 继续"
        )
    second = f"{continuation}\n\n{remainder}".strip()
    return [part for part in (first, second) if part]


def _strip_markdown(text: str) -> str:
    """把 QQ/Markdown 文本降级为纯文本——纯文本通道(微信/WhatsApp)的兜底校验：
    图片无法渲染时直接省略；去 **加粗**/## 标题/残留 #260px 尺寸提示。"""
    import re
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)                 # 图片
    text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)  # 普通链接
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)                      # 加粗
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)                       # 标题
    text = re.sub(r"\s*#\d+px", "", text)                              # 残留尺寸提示
    return text


def _safe_url_for_filter(url: str) -> str:
    """微信 Node 端的 StreamingMarkdownFilter 会把 URL 里的 _ / * 当强调符吃掉
    （且跨整条消息配对，时灵时不灵），破坏图片链接。这里预编码成 %5F / %2A，
    服务端会自动解码回来（已验证 CDN 200），从而绕过过滤器、保证图片 URL 完整。"""
    return str(url or "").replace("_", "%5F").replace("*", "%2A")


def _recipe_collection_cta(recipe_count: int, lang: str) -> str:
    """为整组候选只生成一个下一步，避免每张卡片重复客服式 CTA。"""
    if recipe_count <= 1:
        return (
            'View details: reply "details 1".'
            if lang == "en"
            else "查看详情：回复「详情 1」"
        )
    return (
        'Want the full recipe? Reply with "details" and a number, such as "details 2".'
        if lang == "en"
        else "想看哪道的完整做法，回复「详情 + 序号」，比如「详情 2」。"
    )


def _json_to_plaintext(data: dict, *, encode_filter_urls: bool = True) -> str:
    """纯文本通道渲染：无 markdown，图片转干净链接行，标签按响应语言过滤。
    recipe_search 单独干净渲染并**校验完整性**；其余类型复用 markdown 渲染再降级兜底。"""
    resp_data = data.get("data", {})
    if data.get("type") == "web_search":
        return str(data.get("message") or "").strip()
    if data.get("type") in {"recipe_search", "menu_plan"} and resp_data.get("recipes"):
        lang = _response_lang(data)
        parts = []
        reasoning = _recipe_reasoning(data)
        opening = str((resp_data or {}).get("opening") or "").strip()
        strategy = str((resp_data or {}).get("strategy") or "").strip()
        if data.get("type") == "menu_plan" and not opening:
            opening = str(data.get("message") or "").strip()
        if opening:
            parts.append(opening)
        if strategy:
            parts.append(strategy)
        if opening or strategy:
            parts.append("")
        elif reasoning:
            parts.extend([reasoning, ""])
        if not (opening or strategy):
            parts.append(f"{_recipe_text(lang, 'found_header')}\n")
        for i, recipe in enumerate(resp_data["recipes"], 1):
            if not isinstance(recipe, dict):
                logger.warning(
                    "[plaintext] 第 %s 条食谱结构非法，已跳过：value_type=%s",
                    i, type(recipe).__name__,
                )
                continue
            name = str(recipe.get("name") or "").strip()
            image = str(recipe.get("image") or "").strip()
            ingredients = recipe.get("ingredients") or []

            # 完整性校验：菜名/食材/标签/图片 应保持完整，缺则告警（绝不静默丢弃）
            missing = [label for label, val in
                       (("菜名", name), ("食材", ingredients), ("标签", recipe.get("tags")), ("图片", image))
                       if not val]
            if missing:
                logger.warning("[plaintext] 食谱 #%s 字段缺失：%s", i, missing)

            # 菜名永远输出（【】让它在纯文本里突出），即使为空也兜底占位
            recommendation_reason = str(recipe.get("recommendation_reason") or "").strip()
            line = f"{i}. 【{name or _recipe_text(lang, 'unknown_dish')}】"
            role_label = _menu_role_label(recipe, lang)
            if role_label:
                line += f"（{role_label}）"
            if recommendation_reason:
                line += (": " if lang == "en" else "：") + recommendation_reason
            parts.append(line)
            if image:
                display_image = _safe_url_for_filter(image) if encode_filter_urls else image
                parts.append(f"   🖼️ {_recipe_text(lang, 'image')}: {display_image}")
            ingredient_sentence = _natural_ingredient_sentence(ingredients, lang)
            if ingredient_sentence:
                parts.append(f"   {ingredient_sentence}")
            tags = [
                str(tag).strip()
                for tag in (recipe.get("tags") or [])
                if str(tag).strip()
            ][:6]
            if tags:
                parts.append(
                    f"   Tags: {' · '.join(tags)}"
                    if lang == "en" else f"   标签：{' · '.join(tags)}"
                )
            detail_reference, _ = _recipe_detail_reference(recipe, lang)
            if detail_reference:
                parts.append(
                    f"   Cooking reference: {detail_reference}"
                    if lang == "en" else f"   做饭参考：{detail_reference}"
                )
            parts.append("")
        closing = str((resp_data or {}).get("closing") or "").strip()
        parts.append(
            closing or _recipe_collection_cta(len(resp_data["recipes"]), lang)
        )
        return "\n".join(parts)
    return _strip_markdown(_json_to_markdown(data))


def _recipe_search_delivery(data: dict) -> dict | None:
    """把结构化搜索响应拆成导语、Markdown 卡片和纯文本通道卡片。"""
    resp_data = data.get("data") if isinstance(data, dict) else {}
    recipes = (resp_data or {}).get("recipes") or []
    if data.get("type") not in {"recipe_search", "menu_plan"} or not recipes:
        return None

    lang = _response_lang(data)
    opening = str((resp_data or {}).get("opening") or "").strip()
    strategy = str((resp_data or {}).get("strategy") or "").strip()
    if data.get("type") == "menu_plan" and not opening:
        opening = str(data.get("message") or "").strip()
    reasoning = _recipe_reasoning(data)
    title = (
        "## 🍽️ Menu plan" if lang == "en" else "## 🍽️ 推荐菜单"
    ) if data.get("type") == "menu_plan" else (
        "## 🍳 Recipe ideas" if lang == "en" else "## 🍳 推荐菜谱"
    )
    preface_parts = []
    if opening:
        preface_parts.append(opening)
    elif reasoning:
        preface_parts.append(reasoning)
    else:
        preface_parts.append(_recipe_text(lang, "found_header"))
    preface = "\n\n".join(preface_parts)

    cards = []
    for index, recipe in enumerate(recipes, 1):
        if not isinstance(recipe, dict):
            logger.warning(
                "[channel-card] 第 %s 条食谱结构非法，已跳过：value_type=%s",
                index, type(recipe).__name__,
            )
            continue
        name = str(recipe.get("name") or _recipe_text(lang, "unknown_dish")).strip()
        ingredients = recipe.get("ingredients") or []
        tags = [
            str(tag).strip()
            for tag in (recipe.get("tags") or [])
            if str(tag).strip()
        ][:6]
        reason = str(recipe.get("recommendation_reason") or "").strip()
        first_line = f"{index}. {name}"
        role_label = _menu_role_label(recipe, lang)
        if role_label:
            first_line += f"（{role_label}）"
        lines = [first_line]
        visible_ingredients = [
            str(item).strip() for item in ingredients if str(item).strip()
        ][:8]
        if visible_ingredients:
            value = (
                ", ".join(visible_ingredients)
                if lang == "en" else "、".join(visible_ingredients)
            )
            lines.append(
                f"Main ingredients: {value}"
                if lang == "en" else f"主要食材：{value}"
            )
        if tags:
            lines.append(
                f"Tags: {' · '.join(tags)}"
                if lang == "en" else f"标签：{' · '.join(tags)}"
            )
        if reason:
            lines.append(
                f"Why it fits: {reason}"
                if lang == "en" else f"推荐理由：{reason}"
            )
        detail_reference, _ = _recipe_detail_reference(recipe, lang)
        if detail_reference:
            lines.append(
                f"Cooking reference: {detail_reference}"
                if lang == "en" else f"做饭参考：{detail_reference}"
            )
        markdown_lines = [
            f"**{index}. {name}**",
            "",
            "{{QQ_RECIPE_IMAGE}}",
            "",
        ]
        if role_label:
            markdown_lines.append(
                f"**Menu role**: {role_label}"
                if lang == "en" else f"**菜单定位**： {role_label}"
            )
        if visible_ingredients:
            markdown_lines.append(
                f"**Main ingredients**: {value}"
                if lang == "en" else f"**主要食材**： {value}"
            )
        if tags:
            markdown_lines.append(
                f"**Tags**: {' · '.join(tags)}"
                if lang == "en" else f"**标签**： {' · '.join(tags)}"
            )
        if reason:
            markdown_lines.append(
                f"**Why it fits**: {reason}"
                if lang == "en" else f"**推荐理由**： {reason}"
            )
        if detail_reference:
            markdown_lines.append(
                f"**Cooking reference**: {detail_reference}"
                if lang == "en" else f"**做饭参考**： {detail_reference}"
            )
        cards.append({
            "index": index,
            "name": name,
            "image": str(recipe.get("image") or "").strip(),
            "text": "\n".join(lines),
            "markdown": re.sub(
                r"\n{3,}",
                "\n\n",
                "\n".join(markdown_lines),
            ).strip(),
        })

    closing = str((resp_data or {}).get("closing") or "").strip()
    closing = closing or _recipe_collection_cta(len(recipes), lang)
    closing_parts = []
    if strategy or closing:
        closing_parts.append("## How to choose" if lang == "en" else "## 怎么选")
        if strategy:
            closing_parts.append(strategy)
        if closing:
            closing_parts.append(closing)
    return {
        "lang": lang,
        "title": title,
        "preface": preface,
        # WhatsApp/微信继续使用一个完整导语；QQ 会把自然承接和推荐清单分开发送。
        "intro": "\n\n".join(part for part in (title, preface) if part),
        "cards": cards,
        "closing": "\n\n".join(closing_parts),
    }


def _recipe_image_host_allowed(hostname: str) -> bool:
    """限制服务端代下载图片的域名，避免菜谱元数据被利用发起内网请求。"""
    allowed = {
        item.strip().lower()
        for item in os.getenv(
            "CHANNEL_RECIPE_IMAGE_ALLOWED_HOSTS",
            "images.example.invalid,images.example.invalid,images.example.invalid,"
            "cloudkit-prod.oss-cn-hangzhou.aliyuncs.com,"
            "cloudkit-prod.oss-accelerate.aliyuncs.com,"
            "kitcloudbuckettest1.s3.us-west-1.amazonaws.com",
        ).split(",")
        if item.strip()
    }
    hostname = str(hostname or "").strip().lower().rstrip(".")
    return bool(hostname) and any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in allowed
    )


async def _download_recipe_image(url: str) -> str:
    """安全下载微信待上传图片到临时文件；调用方负责删除。"""
    import tempfile
    from pathlib import Path
    from urllib.parse import urlparse

    import httpx

    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or not _recipe_image_host_allowed(parsed.hostname or ""):
        raise ValueError("菜谱图片 URL 不在允许的 HTTPS 域名范围内")

    try:
        max_bytes = max(1024, int(os.getenv("CHANNEL_RECIPE_IMAGE_MAX_BYTES", "10485760")))
    except (TypeError, ValueError):
        max_bytes = 10 * 1024 * 1024

    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    chunks: list[bytes] = []
    size = 0
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            final = urlparse(str(response.url))
            if final.scheme != "https" or not _recipe_image_host_allowed(final.hostname or ""):
                raise ValueError("菜谱图片重定向到了未授权域名")
            content_type = str(response.headers.get("content-type") or "").lower()
            if content_type and not content_type.startswith("image/"):
                raise ValueError(f"菜谱图片响应类型非法：{content_type}")
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("菜谱图片超过允许大小")
                chunks.append(chunk)

    with tempfile.NamedTemporaryFile(
        prefix="cookclaw-recipe-", suffix=suffix, delete=False
    ) as handle:
        handle.write(b"".join(chunks))
        return handle.name


async def _send_qq_recipe_search(
    event,
    data: dict,
    *,
    lead: str = "",
) -> str | None:
    """把整组推荐作为一条 QQ Markdown 发送，图片内嵌在对应菜名下面。"""
    from urllib.parse import urlparse

    delivery = _recipe_search_delivery(data)
    if not delivery or not _qq_adapter:
        return None

    cards = delivery["cards"]
    if not cards:
        return None

    # QQ 优先保证一条完整 Markdown：标题、唯一导语、卡片和下一步都在同一条，
    # 不再丢掉回复层已经校验过的 opening。
    parts = [
        str(delivery.get("title") or "").strip(),
        str(delivery.get("preface") or "").strip(),
    ]

    embedded_images = 0
    for card in cards:
        markdown = str(card.get("markdown") or card["text"] or "").strip()
        image = str(card.get("image") or "").strip()
        parsed = urlparse(image) if image else None
        safe_image = (
            image
            if parsed
            and parsed.scheme == "https"
            and _recipe_image_host_allowed(parsed.hostname or "")
            else ""
        )
        if safe_image:
            image_markdown = (
                f"![{card['name']} #260px #195px]({safe_image})"
            )
            markdown = markdown.replace("{{QQ_RECIPE_IMAGE}}", image_markdown)
            embedded_images += 1
        elif image:
            logger.warning(
                "[QQBot] 菜谱 #%s Markdown 图片域名不在白名单，已省略",
                card["index"],
            )
            markdown = markdown.replace("{{QQ_RECIPE_IMAGE}}", "")
        else:
            markdown = markdown.replace("{{QQ_RECIPE_IMAGE}}", "")
        parts.append(re.sub(r"\n{3,}", "\n\n", markdown).strip())

    closing = str(delivery["closing"] or "").strip()
    if closing:
        parts.append(closing)

    document = "\n\n".join(part for part in parts if part).strip()
    result = await _qq_adapter.send(
        event.source.chat_id,
        document,
        reply_to=event.message_id,
    )
    if not result.success:
        logger.error("[QQBot] 单条 Markdown 推荐发送失败")
        return None

    logger.info(
        "[QQBot] 单条 Markdown 推荐发送完成：cards=%s embedded_images=%s chars=%s",
        len(cards),
        embedded_images,
        len(document),
    )
    return document


async def _send_whatsapp_recipe_search(event, data: dict) -> str:
    """WhatsApp：导语 + 每道菜原生图片/说明 + 结尾，逐条失败可降级。"""
    delivery = _recipe_search_delivery(data)
    if not delivery or not _wa_adapter:
        return _json_to_plaintext(data, encode_filter_urls=False)

    sent = 0
    if delivery["intro"]:
        await _wa_adapter.send_message(
            to=event.from_jid,
            text=delivery["intro"],
            quoted_message_id=event.message_id or None,
        )
        sent += 1

    for card in delivery["cards"]:
        image = card["image"]
        try:
            await _wa_adapter.send_message(
                to=event.from_jid,
                text=card["text"],
                media_url=image or None,
            )
            sent += 1
        except Exception as exc:
            logger.warning(
                "WhatsApp 菜谱 #%s 图片发送失败，降级文本：error_type=%s",
                card["index"], type(exc).__name__,
            )
            fallback = card["text"]
            if image:
                fallback += f"\n🖼️ {_recipe_text(delivery['lang'], 'image')}: {image}"
            await _wa_adapter.send_message(to=event.from_jid, text=fallback)
            sent += 1

    if delivery["closing"]:
        await _wa_adapter.send_message(to=event.from_jid, text=delivery["closing"])
        sent += 1
    if not sent:
        raise RuntimeError("WhatsApp 菜谱图文消息均未发送")
    return _json_to_plaintext(data, encode_filter_urls=False)


async def _send_weixin_recipe_search(event, data: dict) -> str:
    """微信：菜谱文字与图片分开发送，避免媒体超时后重复投递同一张卡片。"""
    delivery = _recipe_search_delivery(data)
    if not delivery or not _wx_adapter:
        return _json_to_plaintext(data)

    sent = 0
    if delivery["intro"]:
        await _wx_adapter.send_message(
            to=event.from_user_id,
            text=delivery["intro"],
            context_token=event.context_token,
            account_id=event.account_id,
        )
        sent += 1

    for card in delivery["cards"]:
        image = card["image"]
        # Node sidecar 的媒体接口会先发文字、再上传并发送图片。图片较慢时，
        # 文字可能已经送达，但 Python 会因等待整个请求超时而再次降级发文字。
        # 因此这里先独立发送一次卡片文字，后续媒体请求只负责图片。
        await _wx_adapter.send_message(
            to=event.from_user_id,
            text=card["text"],
            context_token=event.context_token,
            account_id=event.account_id,
        )
        sent += 1
        if not image:
            continue

        temp_path = ""
        try:
            temp_path = await _download_recipe_image(image)
        except Exception as exc:
            logger.warning(
                "微信菜谱 #%s 图片下载失败，保留已发送文字并提供图片链接：error_type=%s",
                card["index"], type(exc).__name__,
            )
            await _wx_adapter.send_message(
                to=event.from_user_id,
                text=(
                    f"🖼️ {_recipe_text(delivery['lang'], 'image')}: "
                    f"{_safe_url_for_filter(image)}"
                ),
                context_token=event.context_token,
                account_id=event.account_id,
            )
            sent += 1
            continue

        try:
            await _wx_adapter.send_media(
                to=event.from_user_id,
                file_path=temp_path,
                text="",
                context_token=event.context_token,
                account_id=event.account_id,
            )
            sent += 1
        except Exception as exc:
            logger.warning(
                "微信菜谱 #%s 图片发送失败或状态未知，不重复发送菜谱文字：error_type=%s",
                card["index"], type(exc).__name__,
            )
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError as exc:
                    logger.warning(
                        "清理微信菜谱临时图片失败：error_type=%s",
                        type(exc).__name__,
                    )

    if delivery["closing"]:
        await _wx_adapter.send_message(
            to=event.from_user_id,
            text=delivery["closing"],
            context_token=event.context_token,
            account_id=event.account_id,
        )
        sent += 1
    if not sent:
        raise RuntimeError("微信菜谱图文消息均未发送")
    return _json_to_plaintext(data)


def _image_input_from_media(media: str) -> dict | None:
    """通道媒体路径/URL -> vision 输入。公网 URL 直接传；本地缓存文件转 base64。"""
    import base64
    import mimetypes

    if isinstance(media, str) and media.lower().startswith(("http://", "https://")):
        logger.info("[图片识别] 按公网 URL 处理：origin=%s", _log_url_origin(media))
        return {"image_url": media}
    if isinstance(media, str) and os.path.exists(media):
        mime = mimetypes.guess_type(media)[0] or "image/jpeg"
        with open(media, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logger.info(
            "[图片识别] 按本地缓存文件处理：mime=%s b64_chars=%s",
            mime, len(b64),
        )
        return {"image_base64": b64, "image_mime": mime}
    logger.warning(
        "[图片识别] media 既不是 URL 也不是存在的文件：value_type=%s chars=%s",
        type(media).__name__, len(str(media or "")),
    )
    return None


_SYNTHETIC_IMAGE_TURNS = (
    "[用户发送了",
    "[user sent ",
    "[image]",
    "[图片]",
)


def _recent_conversation_lang(recent_turns) -> str | None:
    """从最近真实用户消息推断图片回复语言，忽略通道写入的图片占位文本。"""
    from app.agent.fast_path import detect_lang

    values: list[str] = []
    for turn in reversed(list(recent_turns or [])[-12:]):
        role = (
            str(turn.get("role") or "")
            if isinstance(turn, dict)
            else str(getattr(turn, "role", "") or "")
        ).lower()
        if role != "user":
            continue
        content = (
            str(turn.get("content") or "")
            if isinstance(turn, dict)
            else str(getattr(turn, "content", "") or "")
        ).strip()
        lowered = content.lower()
        if not content or any(lowered.startswith(prefix) for prefix in _SYNTHETIC_IMAGE_TURNS):
            continue
        values.append(content)
        if len(values) >= 4:
            break
    if not values:
        return None
    # 结合最近几条真实用户消息，避免中文会话中一句 “ok” 就把纯图片回复切成英文。
    return detect_lang(" ".join(reversed(values)))


async def _image_response_lang(chat_id: str, user_text: str = "") -> str:
    """图片消息语言优先级：图文文字 > 最近会话 > 当前候选 > 中文。"""
    from app.agent.fast_path import detect_lang

    if str(user_text or "").strip():
        return detect_lang(user_text)

    try:
        from app.conversation.service import get_conversation_service

        memory = await get_conversation_service().load(chat_id)
        context_lang = _recent_conversation_lang(memory.recent_turns)
        if context_lang:
            logger.info(
                "[图片识别] 回复语言来自最近会话：lang=%s recent_turns=%s",
                context_lang,
                len(memory.recent_turns),
            )
            return context_lang
    except Exception as exc:
        logger.warning(
            "[图片识别] 读取会话语言失败，回退候选语言：error_type=%s",
            type(exc).__name__,
        )

    from app.ports.dialogue_state import recall_candidate_lang

    return recall_candidate_lang(chat_id) or "zh"


async def _image_agent_memory_context(
    chat_id: str,
    current_question: str,
) -> tuple[list, dict]:
    """给图片推荐 Deep Agent 加载近期用户语境和最小结构化任务事实。"""
    from app.agent.controlled_deep_agent import compact_conversation_state
    from app.ports.dialogue_state import snapshot_thread_state

    recent_turns: list = []
    agent_context = compact_conversation_state(snapshot_thread_state(chat_id))
    try:
        from app.conversation.service import get_conversation_service

        context = await get_conversation_service().routing_context(
            chat_id,
            scope="full",
            current_message=current_question,
        )
        recent_turns = list(context.get("recent_turns") or [])
        preferred_name = str(context.get("preferred_name") or "").strip()
        if preferred_name:
            agent_context["user_profile"] = {
                "preferred_name": preferred_name,
                # Phase 2.2: 完整用户画像注入
                "preferences": {
                    "likes": (context.get("preferences") or {}).get("likes", [])[:3],
                    "dislikes": (context.get("preferences") or {}).get("dislikes", [])[:3],
                    "dietary_constraints": (context.get("preferences") or {}).get("dietary_constraints", [])[:3],
                    "allergens": (context.get("preferences") or {}).get("allergens", [])[:2],
                },
            }
    except Exception as exc:
        logger.warning(
            "[图片识别] 读取 Deep Agent 会话语境失败，继续使用本轮视觉事实: error_type=%s",
            type(exc).__name__,
        )
    return recent_turns, agent_context


async def _handle_image_media(
    media_urls: list[str],
    chat_id: str,
    user_text: str = "",
) -> str | dict:
    """通过统一应用服务处理图片消息，并兼容现有通道返回协议。"""
    from app.observability.trace import current_trace
    from app.orchestrator.turn.application_service import TurnExecutionContext
    from app.orchestrator.turn.facade import execute_turn
    from app.orchestrator.turn.image_handler import (
        ImageHandlerResult,
        record_image_turn_observability,
    )
    from app.orchestrator.turn.response_renderer import (
        response_to_envelope,
    )
    from app.orchestrator.turn.runtime_models import (
        ResponseEnvelope,
        TurnRequest,
    )

    owned_trace_token = None
    success = False
    response_type = "image_recognition"
    error_type = None
    channel = str(chat_id or "").split(":", 1)[0].lower()
    if channel not in {"qq", "weixin", "whatsapp", "web"}:
        channel = "unknown"
    trace = current_trace()
    if trace is None:
        _trace, owned_trace_token = ensure_turn_trace(
            channel=channel,
            thread_id=chat_id,
            message_type="image",
            deep_agent_enabled=settings.IMAGE_DEEP_AGENT_ENABLED,
            deep_agent_cohort=(
                "image_required"
                if settings.IMAGE_DEEP_AGENT_ENABLED
                else "image_deterministic"
            ),
        )
        trace = _trace
    request = TurnRequest(
        utterance=str(user_text or ""),
        thread_id=chat_id,
        channel=channel,
        message_type="image",
        trace_id=trace.trace_id if trace is not None else None,
    )
    image_result: ImageHandlerResult | None = None
    compatibility_result: str | dict | None = None

    async def runtime_loader(_context: TurnExecutionContext):
        # 图片领域 handler 使用本轮快照，不需要重复构造文本路由 runtime。
        return None

    async def image_stage(
        context: TurnExecutionContext,
    ) -> ResponseEnvelope:
        nonlocal image_result, compatibility_result
        result = await _handle_image_media_impl(
            request,
            media_urls,
            short_term_snapshot=context.short_term_snapshot,
        )
        if isinstance(result, ImageHandlerResult):
            image_result = result
            return result.envelope

        # 测试替身或自定义扩展可返回现有通道响应；正式实现返回 ImageHandlerResult。
        compatibility_result = result
        if isinstance(result, dict):
            payload = result.get("response")
            if isinstance(payload, dict):
                return response_to_envelope(
                    json.dumps(payload, ensure_ascii=False),
                    handled_by="image_handler",
                    trace_id=request.trace_id,
                )
        return response_to_envelope(
            str(result or ""),
            handled_by="image_handler",
            trace_id=request.trace_id,
        )

    try:
        envelope = await execute_turn(
            request,
            runtime_loader=runtime_loader,
            handlers={"image_handler": image_stage},
        )
        if image_result is not None:
            finalized_result = image_result.with_envelope(envelope)
            result = finalized_result.channel_value()
        else:
            finalized_result = ImageHandlerResult(
                envelope=envelope,
                result_code="IMAGE_COMPATIBILITY_RESULT",
                success=True,
            )
            result = (
                compatibility_result
                if compatibility_result is not None
                else finalized_result.channel_value()
            )
        record_image_turn_observability(finalized_result, request=request)
        if envelope.response_type in {"recipe_search", "menu_plan"}:
            response_type = "image_recipe_search"
        success = True
        return result
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        if owned_trace_token is not None:
            finish_turn_trace(
                owned_trace_token,
                success=success,
                response_type=response_type,
                error_type=error_type,
            )


async def _handle_image_media_impl(
    request,
    media_urls: list[str],
    short_term_snapshot=None,
):
    """图片领域适配器；由应用服务的 image_handler stage 调用。"""
    from app.orchestrator.turn.image_handler import handle_image_turn

    return await handle_image_turn(
        media_urls,
        request=request,
        image_input_converter=_image_input_from_media,
        language_resolver=_image_response_lang,
        memory_loader=_image_agent_memory_context,
        short_term_snapshot=short_term_snapshot,
    )


async def _handle_image_message(event) -> str | dict:
    """QQ 图片消息包装：支持多图、图片文字和结构化菜谱响应。"""
    from app.conversation.service import build_qq_thread_id
    thread_id = build_qq_thread_id(
        event.source.chat_type,
        event.source.chat_id,
        event.source.user_id,
    )
    return await _handle_image_media(
        event.media_urls,
        chat_id=thread_id,
        user_text=event.text or "",
    )

async def _qqbot_message_handler(event):
    """QQ Bot 消息回调 — 收到消息后打印详情"""
    # QQ 消息进入业务层的入口。
    # 处理顺序：
    # 1. 打印原始信息
    # 2. 处理纯图片兜底
    # 3. 调用 Agent
    # 4. 格式化并回发
    import time as _time
    from datetime import datetime
    _handler_start = _time.monotonic()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_message_type = getattr(event.message_type, "value", str(event.message_type)).lower()
    if source_message_type == "voice":
        raw_voice_text = event.text
        event.text = normalize_voice_control_text(event.text)
        if event.text != raw_voice_text:
            logger.info(
                "QQ 语音控制语句标准化: input_chars=%s output_chars=%s",
                len(raw_voice_text or ""), len(event.text or ""),
            )

    print("\n" + "=" * 60)
    print(f"  📨 QQ Bot 收到新消息  [{now}]")
    print(f"  ──────────────────────────────────────")
    print(f"  聊天类型: {event.source.chat_type}")
    print(f"  聊天 ID:  {_log_identifier(event.source.chat_id)}")
    print(f"  发送者:   {_log_identifier(event.source.user_id)}")
    print(f"  消息 ID:  {_log_identifier(event.message_id)}")
    print(f"  消息类型: {event.message_type.value}")
    if event.media_urls:
        print(f"  媒体数量: {len(event.media_urls)}")
    print(f"  ──────────────────────────────────────")
    print(f"  内容: {_log_content(event.text)}")
    print("=" * 60 + "\n")

    # 图片/图文消息：多图识别食材或菜品 + 真实检索候选，让用户回复编号执行。
    # 只产出候选，不自动执行；图片带文字时，文字作为检索约束，不再丢图。
    if event.media_urls:
        media_trace_token = None
        media_trace_success = False
        media_trace_error_type = None
        media_response_type = "image_recognition"
        try:
            from app.conversation.service import (
                build_qq_thread_id,
                get_conversation_service,
                should_persist_exchange,
            )

            media_thread_id = build_qq_thread_id(
                event.source.chat_type,
                event.source.chat_id,
                event.source.user_id,
            )
            media_rollout = deep_agent_rollout_decision(media_thread_id)
            _trace, media_trace_token = ensure_turn_trace(
                channel="qq",
                thread_id=media_thread_id,
                message_type=source_message_type,
                started_monotonic=_handler_start,
                deep_agent_enabled=settings.IMAGE_DEEP_AGENT_ENABLED,
                deep_agent_cohort=(
                    "image_required"
                    if settings.IMAGE_DEEP_AGENT_ENABLED
                    else media_rollout.cohort
                ),
            )
            if _qq_adapter:
                conversation = get_conversation_service()
                user_memory_text = (
                    event.text.strip()
                    or f"[用户发送了{len(event.media_urls)}张图片]"
                )
                conversation.enqueue_turn(
                    media_thread_id,
                    "user",
                    user_memory_text,
                    channel="qq",
                    user_id=event.source.user_id,
                    promote_account_memory=should_persist_exchange(event.text),
                )
                media_session_saved = await conversation.wait_for_session_writes(
                    media_thread_id
                )
                _trace.add_event(
                    "session_memory_write",
                    channel="qq",
                    role="user",
                    status="saved" if media_session_saved else "unavailable",
                )
                try:
                    image_result = await _handle_image_message(event)
                except Exception as e:
                    logger.error("图片识别处理异常: error_type=%s", type(e).__name__)
                    image_result = "图片识别出了点问题，换张图、或直接打字告诉我有哪些食材吧。"

                reply_text = ""
                if isinstance(image_result, dict):
                    response_data = image_result.get("response")
                    if isinstance(response_data, dict):
                        media_response_type = "image_recipe_search"
                        with trace_stage("channel_send"):
                            reply_text = (
                                await _send_qq_recipe_search(event, response_data)
                                or ""
                            )
                        if not reply_text:
                            reply_text = _json_to_plaintext(
                                response_data,
                                encode_filter_urls=False,
                            )
                            with trace_stage("channel_send"):
                                await _qq_adapter.send(
                                    event.source.chat_id,
                                    reply_text,
                                    reply_to=event.message_id,
                                )
                else:
                    reply_text = str(image_result or "").strip()
                    if reply_text:
                        with trace_stage("channel_send"):
                            await _qq_adapter.send(
                                event.source.chat_id,
                                reply_text,
                                reply_to=event.message_id,
                            )
                if reply_text:
                    conversation.enqueue_turn(
                        media_thread_id,
                        "assistant",
                        reply_text,
                        channel="qq",
                        user_id=event.source.user_id,
                    )
            media_trace_success = True
        except Exception as exc:
            media_trace_error_type = type(exc).__name__
            raise
        finally:
            finish_turn_trace(
                media_trace_token,
                success=media_trace_success,
                response_type=media_response_type,
                error_type=media_trace_error_type,
            )
        return

    # 没有文本就不继续送进 Agent。
    if not event.text.strip():
        return

    trace_token = None
    trace_success = False
    trace_error_type = None
    try:
        from app.conversation.service import (
            build_qq_thread_id,
            get_conversation_service,
            should_persist_exchange,
        )
        thread_id = build_qq_thread_id(
            event.source.chat_type,
            event.source.chat_id,
            event.source.user_id,
        )
        rollout = deep_agent_rollout_decision(thread_id)
        _trace, trace_token = ensure_turn_trace(
            channel="qq",
            thread_id=thread_id,
            message_type=source_message_type,
            started_monotonic=_handler_start,
            deep_agent_enabled=rollout.enabled,
            deep_agent_cohort=rollout.cohort,
        )
        conversation = get_conversation_service()
        promote_account_memory = should_persist_exchange(event.text)
        conversation.enqueue_turn(
            thread_id,
            "user",
            event.text,
            channel="qq",
            user_id=event.source.user_id,
            promote_account_memory=promote_account_memory,
        )
        session_saved = await conversation.wait_for_session_writes(thread_id)
        _trace.add_event(
            "session_memory_write",
            channel="qq",
            role="user",
            status="saved" if session_saved else "unavailable",
        )

        typing_sent = False

        async def _ensure_qq_typing() -> None:
            nonlocal typing_sent
            if typing_sent or not _qq_adapter:
                return
            send_typing = getattr(_qq_adapter, "send_typing", None)
            if send_typing is None:
                return
            try:
                await send_typing(event.source.chat_id)
                typing_sent = True
            except Exception as exc:
                logger.debug("QQ 输入状态发送失败: error_type=%s", type(exc).__name__)

        # 先给用户可见反馈，再进行意图、记忆和检索操作。
        await _ensure_qq_typing()

        async def _qq_search_progress(search_query: str, lang: str, bridge: str = "") -> None:
            """搜索期间只显示输入状态；自然承接合并到最终菜谱卡片顶部。"""
            await _ensure_qq_typing()

        # 交给 Agent 做意图识别、搜索或问答。
        raw_response = await qqbot_chat(
            event.text,
            thread_id=thread_id,
            on_search_start=_qq_search_progress,
            source_message_type=source_message_type,
        )

        if not raw_response.strip():
            trace_success = True
            return

        # 如果 Agent 返回 JSON，就把它转成更适合 QQ 展示的 Markdown。
        reply_text = raw_response
        reply_messages = [raw_response]
        recipe_media_sent = False
        try:
            import json
            # 剥离 AI 可能包裹的 markdown 代码块（```json ... ```）
            json_str = raw_response.strip()
            if json_str.startswith("```"):
                # 移除开头的 ```json 或 ```
                first_newline = json_str.index("\n") if "\n" in json_str else len(json_str)
                json_str = json_str[first_newline + 1:]
                # 移除结尾的 ```
                if json_str.rstrip().endswith("```"):
                    json_str = json_str.rstrip()[:-3].rstrip()
            data = json.loads(json_str)
            if isinstance(data, dict):
                # 仅记录响应形状，避免把查询、用户偏好和菜谱正文写入日志。
                print(f"  📋 JSON 响应: type={data.get('type')}, intent={data.get('intent')}")
                payload_data = data.get("data")
                if isinstance(payload_data, dict):
                    recipes_count = len(payload_data.get("recipes") or [])
                    print(
                        f"  📊 数据结构: keys={sorted(payload_data.keys())}, "
                        f"recipes={recipes_count}"
                    )

                # QQ 菜谱图走文件上传 + 原生媒体消息，避开不稳定的外链图片代理。
                # 其它响应继续使用单条 Markdown。
                if (
                    data.get("type") in {"recipe_search", "menu_plan"}
                    and isinstance(payload_data, dict)
                    and payload_data.get("recipes")
                ):
                    with trace_stage("channel_send"):
                        delivered_text = await _send_qq_recipe_search(event, data)
                    if delivered_text is not None:
                        recipe_media_sent = True
                        reply_text = delivered_text
                        reply_messages = []
                    else:
                        # 原生媒体整体不可用时只发纯文本，绝不回退到破图外链。
                        reply_text = _json_to_plaintext(data)
                        reply_messages = [reply_text]
                else:
                    reply_messages = _qq_markdown_messages(data)
                    reply_text = "\n\n".join(reply_messages)
        except (json.JSONDecodeError, TypeError):
            pass  # 非 JSON 响应，直接发送原文

        # 最终把整理好的文本回发给 QQ。
        if reply_text.strip() and _qq_adapter and not recipe_media_sent:
            with trace_stage("channel_send"):
                for message_index, message_text in enumerate(reply_messages):
                    if not str(message_text or "").strip():
                        continue
                    await _qq_adapter.send(
                        event.source.chat_id,
                        message_text,
                        reply_to=event.message_id if message_index == 0 else None,
                    )
                    # QQ 私信连续发多条存在频率限制，给第二个菜单气泡留出稳定窗口。
                    if message_index < len(reply_messages) - 1:
                        await asyncio.sleep(0.6)

        # 无论最终走纯文本还是原生图片卡片，都保存同一份完整语义，保证用户
        # 后续说“第二道 / 换一批 / 详情 1”时能承接到刚展示的真实菜谱。
        if reply_text.strip():
            conversation.enqueue_turn(
                thread_id,
                "assistant",
                reply_text,
                channel="qq",
                user_id=event.source.user_id,
            )

        _handler_elapsed = _time.monotonic() - _handler_start
        print(f"  ⏱️ 消息处理完成，总耗时: {_handler_elapsed:.2f}s")
        trace_success = True
    except Exception as e:
        trace_error_type = type(e).__name__
        _handler_elapsed = _time.monotonic() - _handler_start
        logger.error("回复消息异常: error_type=%s elapsed=%.2fs", type(e).__name__, _handler_elapsed)
    finally:
        finish_turn_trace(
            trace_token,
            success=trace_success,
            error_type=trace_error_type,
        )

#入口
def _create_qqbot_adapter():
    """从环境变量创建 QQ Bot 适配器"""
    # 这一段负责把 .env 里的变量收集起来，喂给 QQAdapter。
    # 排查 QQ 启动问题时，最先看这里读到了什么。
    from app.qqbot.adapter import QQAdapter

    extra = {
        "app_id": os.getenv("QQ_APP_ID", ""),
        "client_secret": os.getenv("QQ_CLIENT_SECRET", ""),
        "markdown_support": os.getenv("QQ_MARKDOWN_SUPPORT", "true").lower() == "true",
        "dm_policy": os.getenv("QQ_DM_POLICY", "open"),
        "allow_from": os.getenv("QQ_ALLOW_FROM", "").split(",") if os.getenv("QQ_ALLOW_FROM") else [],
        "group_policy": os.getenv("QQ_GROUP_POLICY", "open"),
        "group_allow_from": os.getenv("QQ_GROUP_ALLOW_FROM", "").split(",") if os.getenv("QQ_GROUP_ALLOW_FROM") else [],
    }

    stt_api_key = os.getenv("QQ_STT_API_KEY", "")
    if stt_api_key:
        extra["stt"] = {
            "provider": os.getenv("QQ_STT_PROVIDER", "zai"),
            "baseUrl": os.getenv("QQ_STT_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"),
            "apiKey": stt_api_key,
            "model": os.getenv("QQ_STT_MODEL", "glm-asr"),
        }

    adapter = QQAdapter(extra=extra)
    adapter.set_message_handler(_qqbot_message_handler)
    return adapter


async def _whatsapp_message_handler(event):
    """WhatsApp 消息回调 — 收到消息后调用 Agent 处理并回复"""
    # 结构和 QQ 的回调很像，只是字段名不同。
    import time as _time
    from datetime import datetime
    from app.whatsapp.adapter import WhatsAppMessageEvent

    _handler_start = _time.monotonic()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print(f"  📨 WhatsApp 收到新消息  [{now}]")
    print(f"  ──────────────────────────────────────")
    print(f"  聊天类型: {event.chat_type}")
    print(f"  发送者:   {_log_identifier(event.from_number)}")
    print(f"  消息 ID:  {_log_identifier(event.message_id)}")
    if event.media_types:
        print(f"  媒体类型: {', '.join(event.media_types)}")
    if event.group_jid:
        print(f"  群组 JID: {_log_identifier(event.group_jid)}")
    print(f"  ──────────────────────────────────────")
    print(f"  内容: {_log_content(event.text)}")
    print("=" * 60 + "\n")

    # WhatsApp 会话键必须带通道、聊天和发送者三层信息：
    # 私聊按号码隔离；群聊按“群组 + 成员”隔离，不能让不同成员共享偏好和候选。
    from app.conversation.service import (
        build_whatsapp_thread_id,
        get_conversation_service,
        should_persist_exchange,
    )
    chat_id = event.group_jid or event.from_jid
    user_id = event.participant or event.from_number or event.from_jid
    thread_id = build_whatsapp_thread_id(event.chat_type, chat_id, user_id)
    conversation = get_conversation_service()
    promote_account_memory = should_persist_exchange(event.text)

    async def _remember(role: str, content: str) -> None:
        if not str(content or "").strip():
            return
        conversation.enqueue_turn(
            thread_id,
            role,
            content,
            channel="whatsapp",
            user_id=user_id,
            promote_account_memory=(
                promote_account_memory if role == "user" else False
            ),
        )
        if role == "user":
            saved = await conversation.wait_for_session_writes(thread_id)
            if not saved:
                logger.warning(
                    "WhatsApp 用户轮次未及时写入，会按降级上下文继续"
                )

    # 图片/图文消息：多图识别食材或菜品 + 真实检索候选。
    if event.media_urls:
        if _wa_adapter:
            memory_text = event.text.strip() or f"[用户发送了{len(event.media_urls)}张图片]"
            await _remember("user", memory_text)
            try:
                image_result = await _handle_image_media(
                    event.media_urls,
                    chat_id=thread_id,
                    user_text=event.text or "",
                )
            except Exception as e:
                logger.error("WhatsApp 图片识别处理异常: error_type=%s", type(e).__name__)
                image_result = "图片识别出了点问题，换张图、或直接打字告诉我有哪些食材吧。"
            if isinstance(image_result, dict) and isinstance(
                image_result.get("response"),
                dict,
            ):
                reply_text = await _send_whatsapp_recipe_search(
                    event,
                    image_result["response"],
                )
            else:
                reply_text = str(image_result or "").strip()
                if reply_text:
                    await _wa_adapter.send_message(
                        to=event.from_jid,
                        text=reply_text,
                        quoted_message_id=event.message_id or None,
                    )
            await _remember("assistant", reply_text)
        return

    # 没有文本就结束。
    if not event.text.strip():
        return

    try:
        await _remember("user", event.text)

        async def _wa_search_progress(search_query: str, lang: str, bridge: str = "") -> None:
            if bridge.strip() and _wa_adapter:
                await _wa_adapter.send_message(
                    to=event.from_jid,
                    text=bridge.strip(),
                    quoted_message_id=event.message_id or None,
                )

        # 复用共享对话策略；缓冲消息会在真实检索期间先送达。
        raw_response = await qqbot_chat(
            event.text,
            thread_id=thread_id,
            on_search_start=_wa_search_progress,
        )

        if not raw_response.strip():
            return

        # 纯文本通道：JSON → 纯文本（无 markdown，图片转链接行）。
        reply_text = raw_response
        recipe_search_delivered = False
        try:
            json_str = raw_response.strip()
            if json_str.startswith("```"):
                first_newline = json_str.index("\n") if "\n" in json_str else len(json_str)
                json_str = json_str[first_newline + 1:]
                if json_str.rstrip().endswith("```"):
                    json_str = json_str.rstrip()[:-3].rstrip()
            data = json.loads(json_str)
            if isinstance(data, dict):
                if _recipe_search_delivery(data):
                    reply_text = await _send_whatsapp_recipe_search(event, data)
                    recipe_search_delivered = True
                else:
                    reply_text = _json_to_plaintext(data, encode_filter_urls=False)
        except (json.JSONDecodeError, TypeError):
            pass  # 非 JSON 响应，直接发送原文

        # 发回 WhatsApp。
        if reply_text.strip() and _wa_adapter and not recipe_search_delivered:
            await _wa_adapter.send_message(
                to=event.from_jid,
                text=reply_text,
                quoted_message_id=event.message_id or None,
            )
        if reply_text.strip():
            await _remember("assistant", reply_text)

        _handler_elapsed = _time.monotonic() - _handler_start
        print(f"  ⏱️ WhatsApp 消息处理完成，总耗时: {_handler_elapsed:.2f}s")
    except Exception as e:
        _handler_elapsed = _time.monotonic() - _handler_start
        logger.error(
            "WhatsApp 回复消息异常: error_type=%s elapsed=%.2fs",
            type(e).__name__, _handler_elapsed,
        )


async def _weixin_message_handler(event):
    """微信消息回调 — 收到消息后调用 Agent 处理并回复"""
    # 微信通道与 QQ/WhatsApp 的处理骨架一致：
    # 先打印，再做媒体兜底，再交给 Agent。
    import time as _time
    from datetime import datetime

    _handler_start = _time.monotonic()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print(f"  📨 微信收到新消息  [{now}]")
    print(f"  ──────────────────────────────────────")
    print(f"  发送者:   {_log_identifier(event.from_user_id)}")
    print(f"  消息 ID:  {_log_identifier(event.message_id)}")
    if event.image_path:
        print("  图片:     [redacted path]")
    if event.voice_path:
        print("  语音:     [redacted path]")
    if event.file_path:
        print("  文件:     [redacted path]")
    if event.video_path:
        print("  视频:     [redacted path]")
    print(f"  ──────────────────────────────────────")
    print(f"  内容: {_log_content(event.text)}")
    print("=" * 60 + "\n")

    # 与 WhatsApp 对齐：微信会话键包含通道、机器人账号和用户三层信息，
    # 并把用户/助手轮次写入同一套共享短期记忆。
    from app.conversation.service import (
        build_weixin_thread_id,
        get_conversation_service,
        should_persist_exchange,
    )
    user_id = event.from_user_id
    thread_id = build_weixin_thread_id(event.account_id, user_id)
    conversation = get_conversation_service()
    promote_account_memory = should_persist_exchange(event.text)

    async def _remember(role: str, content: str) -> None:
        if not str(content or "").strip():
            return
        conversation.enqueue_turn(
            thread_id,
            role,
            content,
            channel="weixin",
            user_id=user_id,
            promote_account_memory=(
                promote_account_memory if role == "user" else False
            ),
        )
        if role == "user":
            saved = await conversation.wait_for_session_writes(thread_id)
            if not saved:
                logger.warning(
                    "微信用户轮次未及时写入，会按降级上下文继续"
                )

    # 图片/图文消息：识别食材或菜品 + 真实检索候选。
    if event.image_path:
        if _wx_adapter:
            memory_text = event.text.strip() or "[用户发送了1张图片]"
            await _remember("user", memory_text)
            try:
                image_result = await _handle_image_media(
                    [event.image_path],
                    chat_id=thread_id,
                    user_text=event.text or "",
                )
            except Exception as e:
                logger.error("微信图片识别处理异常: error_type=%s", type(e).__name__)
                image_result = "图片识别出了点问题，换张图、或直接打字告诉我有哪些食材吧。"
            if isinstance(image_result, dict) and isinstance(
                image_result.get("response"),
                dict,
            ):
                reply_text = await _send_weixin_recipe_search(
                    event,
                    image_result["response"],
                )
            else:
                reply_text = str(image_result or "").strip()
                if reply_text:
                    await _wx_adapter.send_message(
                        to=event.from_user_id,
                        text=reply_text,
                        context_token=event.context_token,
                        account_id=event.account_id,
                    )
            await _remember("assistant", reply_text)
        return

    # 其他纯媒体消息暂不识别，先明确回复用户。
    if not event.text.strip() and (event.voice_path or event.file_path or event.video_path):
        if _wx_adapter:
            await _wx_adapter.send_message(
                to=event.from_user_id,
                text="我收到了这条媒体消息，但当前还不支持识别这种类型。你可以补充一句文字描述，我再帮你继续处理。",
                context_token=event.context_token,
                account_id=event.account_id,
            )
        return

    # 没有文本就结束。
    if not event.text.strip():
        return

    try:
        await _remember("user", event.text)

        async def _wx_search_progress(search_query: str, lang: str, bridge: str = "") -> None:
            if bridge.strip() and _wx_adapter:
                await _wx_adapter.send_message(
                    to=event.from_user_id,
                    text=bridge.strip(),
                    context_token=event.context_token,
                    account_id=event.account_id,
                )

        # 复用共享对话策略；缓冲消息会在真实检索期间先送达。
        raw_response = await qqbot_chat(
            event.text,
            thread_id=thread_id,
            on_search_start=_wx_search_progress,
        )

        if not raw_response.strip():
            return

        # 纯文本通道：解析 JSON 后转纯文本（无 markdown，图片转链接行）。
        reply_text = raw_response
        recipe_search_delivered = False
        try:
            json_str = raw_response.strip()
            if json_str.startswith("```"):
                first_newline = json_str.index("\n") if "\n" in json_str else len(json_str)
                json_str = json_str[first_newline + 1:]
                if json_str.rstrip().endswith("```"):
                    json_str = json_str.rstrip()[:-3].rstrip()
            data = json.loads(json_str)
            if isinstance(data, dict):
                if _recipe_search_delivery(data):
                    reply_text = await _send_weixin_recipe_search(event, data)
                    recipe_search_delivered = True
                else:
                    reply_text = _json_to_plaintext(data)
        except (json.JSONDecodeError, TypeError):
            pass  # 非 JSON 响应，直接发送原文

        # 发回微信。
        if reply_text.strip() and _wx_adapter and not recipe_search_delivered:
            await _wx_adapter.send_message(
                to=event.from_user_id,
                text=reply_text,
                context_token=event.context_token,
                account_id=event.account_id,
            )
        if reply_text.strip():
            await _remember("assistant", reply_text)

        _handler_elapsed = _time.monotonic() - _handler_start
        print(f"  ⏱️ 微信消息处理完成，总耗时: {_handler_elapsed:.2f}s")
    except Exception as e:
        _handler_elapsed = _time.monotonic() - _handler_start
        logger.error(
            "微信回复消息异常: error_type=%s elapsed=%.2fs",
            type(e).__name__, _handler_elapsed,
        )


async def _monitor_conversation_storage(service) -> None:
    """运行期协议级检查 Redis/PG；恢复后提前关闭对应熔断。"""
    interval = max(
        5.0,
        float(os.getenv("CONVERSATION_HEALTHCHECK_INTERVAL_SECONDS", "15")),
    )
    timeout = max(
        0.5,
        float(os.getenv("CONVERSATION_HEALTHCHECK_TIMEOUT_SECONDS", "10")),
    )
    while True:
        await asyncio.sleep(interval)
        started = asyncio.get_running_loop().time()
        try:
            health = await asyncio.wait_for(service.healthcheck(), timeout=timeout)
            if not all(health.values()):
                logger.warning(
                    "会话存储运行期检查失败: health=%s elapsed_ms=%.1f",
                    health,
                    (asyncio.get_running_loop().time() - started) * 1000,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "会话存储运行期检查异常: elapsed_ms=%.1f error_type=%s",
                (asyncio.get_running_loop().time() - started) * 1000,
                type(exc).__name__,
            )


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """FastAPI 生命周期管理 — 启动/关闭 QQ Bot、WhatsApp 和微信"""
    # FastAPI 启动时进入这里，退出时也会回到这里的 yield 之后。
    # 所有平台子进程、连接、任务都在这里统一管理。
    global _qq_adapter, _qq_task, _conversation_health_task, _im_graph_bridge
    global _wa_adapter, _wa_subprocess, _wx_adapter, _wx_subprocess

    # ─── 检查共享记忆存储 ────────────────────────────────────────
    # Redis/PG 模式默认启动即校验，避免通道已经收消息后才发现记忆写入失败。
    from app.conversation.service import get_conversation_service
    _conversation_service = get_conversation_service()
    try:
        _storage_health = await asyncio.wait_for(
            _conversation_service.healthcheck(),
            timeout=float(os.getenv("CONVERSATION_STORAGE_HEALTH_TIMEOUT_SECONDS", "10")),
        )
        if not all(_storage_health.values()):
            raise RuntimeError("conversation storage healthcheck returned false")
        logger.info(
            "会话存储已就绪: conversation=%s profile=%s",
            os.getenv("CONVERSATION_STORE", "redis"),
            os.getenv("PROFILE_STORE", "postgres"),
        )
    except Exception:
        if os.getenv("CONVERSATION_STORAGE_REQUIRED", "true").lower() == "true":
            raise
        logger.exception("会话存储检查失败，按配置继续启动")
    _conversation_health_task = asyncio.create_task(
        _monitor_conversation_storage(_conversation_service)
    )

    # ─── 预热食谱检索（进程内常驻服务，route A）─────────────────
    # 启动时建立 Milvus 连接 / load 集合 / embedding·rerank client，常驻复用，
    # 使首个真实请求即命中热实例。非致命：失败/超时仅告警，运行时自动回退子进程。
    try:
        from app.agent.recipe_search_service import warmup as _recipe_warmup
        await asyncio.wait_for(_recipe_warmup(), timeout=45)
    except asyncio.TimeoutError:
        logger.warning("食谱检索预热超时（>45s），忽略，运行时回退子进程")
    except Exception as e:
        logger.warning("食谱检索预热异常（忽略，运行时回退子进程）：error_type=%s", type(e).__name__)

    # ─── 可选的三 Agent Graph bridge ───────────────────────────
    # 只在显式灰度开关打开时注入 QQ/微信/WhatsApp 的共享入口。
    if os.getenv("MULTI_AGENT_BRIDGE_ENABLED", "false").lower() == "true":
        from pathlib import Path

        from app.agent import participle_agent as _agent_module
        from app.demo.im_bridge import IMGraphBridge

        default_dir = Path(__file__).resolve().parents[1] / ".demo"
        data_dir = Path(
            os.getenv("MULTI_AGENT_DATA_DIR", str(default_dir))
        ).expanduser()
        if not (data_dir / "recipes.db").exists():
            raise RuntimeError(
                "MULTI_AGENT_BRIDGE_ENABLED=true，但公开演示食谱库不存在；"
                "请先运行 python -m app.demo.seed"
            )
        _im_graph_bridge = await IMGraphBridge.create(data_dir)
        _agent_module.bind_im_graph_bridge(_im_graph_bridge)
        logger.info(
            "三 Agent Graph bridge 已启用: mode=%s channels=qq,weixin,whatsapp",
            os.getenv("MULTI_AGENT_BRIDGE_MODE", "live"),
        )
    else:
        logger.info("三 Agent Graph bridge 未启用")

    # ─── 启动 QQ Bot ────────────────────────────────────────────
    # QQ Bot 是 Python 进程内直接连 QQ 网关，不需要额外 Node 服务。
    qq_enabled = os.getenv("QQ_BOT_ENABLED", "false").lower() == "true"
    if qq_enabled:
        _qq_adapter = _create_qqbot_adapter()

        if not _qq_adapter._app_id or _qq_adapter._app_id == "your-app-id":
            logger.warning("QQ Bot 已启用但未配置 QQ_APP_ID，跳过启动")
        else:
            _qq_task = asyncio.create_task(_qq_adapter.run())
            logger.info("QQ Bot 后台任务已启动")
    else:
        logger.info("QQ Bot 未启用 (设置 QQ_BOT_ENABLED=true 启用)")

    # ─── 启动 WhatsApp ──────────────────────────────────────────
    # WhatsApp 通过 Node 微服务承接协议层，Python 只负责编排和转发。
    wa_enabled = settings.WHATSAPP_ENABLED
    if wa_enabled:
        import subprocess
        import httpx

        print("\n" + "=" * 60)
        print("  📱 WhatsApp 集成启动中...")
        print("=" * 60)

        # 1. 启动 WhatsApp 微服务子进程
        wa_service_dir = os.path.join(os.path.dirname(__file__), "whatsapp", "service")
        wa_entry = os.path.join(wa_service_dir, "dist", "index.js")

        # 如果没有构建过，先构建
        if not os.path.exists(wa_entry):
            print("  🔨 WhatsApp 微服务未构建，正在构建...")
            build_proc = subprocess.run(
                ["npm", "run", "build"],
                cwd=wa_service_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if build_proc.returncode != 0:
                print(f"  ❌ WhatsApp 微服务构建失败: {build_proc.stderr}")
            else:
                print("  ✅ WhatsApp 微服务构建完成")

        if os.path.exists(wa_entry):
            # 解析 WhatsApp 服务端口
            wa_url = settings.WHATSAPP_SERVICE_URL
            wa_port = int(wa_url.split(":")[-1].rstrip("/")) if ":" in wa_url else 3002

            # 子进程通过环境变量拿到运行配置。
            os.environ["PORT"] = str(wa_port)
            os.environ["WEBHOOK_URL"] = f"http://localhost:{settings.PORT}{settings.API_PREFIX}/whatsapp/webhook"
            if settings.WHATSAPP_WEBHOOK_SECRET:
                os.environ["WEBHOOK_SECRET"] = settings.WHATSAPP_WEBHOOK_SECRET
            if settings.WHATSAPP_API_TOKEN:
                os.environ["API_TOKEN"] = settings.WHATSAPP_API_TOKEN

            # 代理配置（从 app/whatsapp/service/.env 读取）
            wa_env_file = os.path.join(wa_service_dir, ".env")
            if os.path.exists(wa_env_file):
                with open(wa_env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("PROXY_URL="):
                            proxy_url = line.split("=", 1)[1].strip()
                            if proxy_url:
                                os.environ["PROXY_URL"] = proxy_url
                                print("  🌐 代理配置已加载（地址与凭据不写入日志）")
                            break

            _wa_subprocess = subprocess.Popen(
                ["node", wa_entry],
                cwd=wa_service_dir,
                stdout=None,  # 直接输出到终端
                stderr=None,
            )
            print(f"  🚀 WhatsApp 微服务子进程已启动 (PID: {_wa_subprocess.pid}, 端口: {wa_port})")

            # 2. 等待微服务就绪
            print(f"  ⏳ 等待微服务就绪...")
            max_wait = 15
            for i in range(max_wait):
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"{wa_url}/health", timeout=2)
                        if resp.status_code == 200:
                            print(f"  ✅ WhatsApp 微服务就绪 ({i + 1}s)")
                            break
                except Exception:
                    pass
            else:
                print(f"  ⚠️ WhatsApp 微服务在 {max_wait}s 内未就绪")

        # 3. 初始化适配器
        from app.whatsapp.adapter import WhatsAppAdapter

        _wa_adapter = WhatsAppAdapter(
            base_url=settings.WHATSAPP_SERVICE_URL,
            api_token=settings.WHATSAPP_API_TOKEN,
        )
        _wa_adapter.set_message_handler(_whatsapp_message_handler)
        await _wa_adapter.start()

        # 4. 配置 Webhook，把微信/WhatsApp 的入站消息转回 Python 主服务。
        webhook_url = f"http://localhost:{settings.PORT}{settings.API_PREFIX}/whatsapp/webhook"
        try:
            await _wa_adapter.configure_webhook(
                url=webhook_url,
                secret=settings.WHATSAPP_WEBHOOK_SECRET or None,
            )
            print(f"  ✅ WhatsApp Webhook 已配置 → {webhook_url}")
        except Exception as e:
            print(f"  ⚠️ WhatsApp Webhook 配置失败: error_type={type(e).__name__}")

        print("  ✅ WhatsApp 适配器已启动")
        print("=" * 60 + "\n")
    else:
        print("  ℹ️ WhatsApp 未启用 (设置 WHATSAPP_ENABLED=true 启用)")

    # ─── 启动微信机器人 ──────────────────────────────────────────
    # 微信同样是 Node 微服务 + Python 回调的模式。
    wx_enabled = settings.WEIXIN_ENABLED
    if wx_enabled:
        import subprocess
        import httpx

        # 登录态：兼容旧单账号文件，同时按 account_id 恢复全部已保存账号。
        wx_sessions = _load_wx_sessions()
        wx_session = _load_wx_session()
        wx_token = wx_session.get("token", "") or settings.WEIXIN_TOKEN

        print("\n" + "=" * 60)
        print("  💬 微信机器人集成启动中...")
        print("=" * 60)

        # 1. 启动微信微服务子进程
        wx_service_dir = os.path.join(os.path.dirname(__file__), "weixinbot")
        wx_entry = os.path.join(wx_service_dir, "dist", "server.js")

        # 如果没有构建过，先构建
        if not os.path.exists(wx_entry):
            print("  🔨 微信微服务未构建，正在构建...")
            build_proc = subprocess.run(
                ["npm", "run", "build"],
                cwd=wx_service_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if build_proc.returncode != 0:
                print(f"  ❌ 微信微服务构建失败: {build_proc.stderr}")
            else:
                print("  ✅ 微信微服务构建完成")

        if os.path.exists(wx_entry):
            wx_url = settings.WEIXIN_SERVICE_URL
            wx_port = int(wx_url.split(":")[-1].rstrip("/")) if ":" in wx_url else 3003

            # 设置 Node 微服务运行所需的环境变量。
            wx_env = os.environ.copy()
            wx_env["PORT"] = str(wx_port)
            wx_env["WEBHOOK_URL"] = f"http://localhost:{settings.PORT}{settings.API_PREFIX}/weixin/webhook"
            wx_env["WEIXIN_MAX_ACCOUNTS"] = str(settings.WEIXIN_MAX_ACCOUNTS)
            if settings.WEIXIN_WEBHOOK_SECRET:
                wx_env["WEBHOOK_SECRET"] = settings.WEIXIN_WEBHOOK_SECRET
            if wx_token:
                wx_env["WEIXIN_TOKEN"] = wx_token
            if wx_session.get("base_url"):
                wx_env["WEIXIN_BASE_URL"] = wx_session["base_url"]
            if wx_session.get("account_id"):
                wx_env["WEIXIN_ACCOUNT_ID"] = wx_session["account_id"]

            _wx_subprocess = subprocess.Popen(
                ["node", wx_entry],
                cwd=wx_service_dir,
                stdout=None,
                stderr=None,
                env=wx_env,
            )
            print(f"  🚀 微信微服务子进程已启动 (PID: {_wx_subprocess.pid}, 端口: {wx_port})")

            # 2. 等待微服务就绪
            print(f"  ⏳ 等待微服务就绪...")
            max_wait = 10
            for i in range(max_wait):
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"{wx_url}/health", timeout=2)
                        if resp.status_code == 200:
                            print(f"  ✅ 微信微服务就绪 ({i + 1}s)")
                            break
                except Exception:
                    pass
            else:
                print(f"  ⚠️ 微信微服务在 {max_wait}s 内未就绪")

        # 3. 初始化适配器
        from app.weixinbot.adapter import WeixinAdapter, WeixinAuthExpiredError

        _wx_adapter = WeixinAdapter(base_url=settings.WEIXIN_SERVICE_URL)
        _wx_adapter.set_message_handler(_weixin_message_handler)

        # 4. 把 Python 备份登录态注册到 Node；Node 自己也会从 ~/.weixinbot 恢复账号。
        for account_id, session in wx_sessions.items():
            try:
                await _wx_adapter.set_token(
                    session.get("token", ""),
                    session.get("base_url", ""),
                    account_id,
                    session.get("user_id", ""),
                )
            except Exception as e:
                print(
                    f"  ⚠️ 微信账号 {_log_identifier(account_id)} 恢复失败："
                    f"error_type={type(e).__name__}"
                )

        try:
            status = await _wx_adapter.get_status()
            if status.account_count:
                await _wx_restart_listening()
                print(
                    f"  ✅ 微信账号监听启动中（已注册 {status.account_count}/"
                    f"{status.max_accounts} 个账号）"
                )
            else:
                print("  ℹ️ 无已保存登录，请通过 /weixin/login 扫码添加微信账号")
        except WeixinAuthExpiredError:
            print("  ⚠️ 微信登录已过期，需要重新扫码：/weixin/login")
        except Exception as e:
            print(f"  ⚠️ 微信监听启动失败：error_type={type(e).__name__}")

        print("=" * 60 + "\n")
    else:
        print("  ℹ️ 微信机器人未启用 (设置 WEIXIN_ENABLED=true 启用)")

    yield

    # ─── 关闭 ──────────────────────────────────────────────────
    # 退出时先关业务适配器，再停子进程，最后断开共享存储。
    if _wx_adapter:
        try:
            await _wx_adapter.stop()
        except Exception:
            pass
        await _wx_adapter.close()
        logger.info("微信适配器已关闭")

    if _wx_subprocess:
        _wx_subprocess.terminate()
        try:
            _wx_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _wx_subprocess.kill()
        logger.info("微信微服务子进程已停止")

    if _wa_adapter:
        await _wa_adapter.stop()
        logger.info("WhatsApp 适配器已关闭")

    if _wa_subprocess:
        _wa_subprocess.terminate()
        try:
            _wa_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _wa_subprocess.kill()
        logger.info("WhatsApp 微服务子进程已停止")

    if _qq_adapter:
        await _qq_adapter.disconnect()
        logger.info("QQ Bot 已断开连接")
    if _qq_task and not _qq_task.done():
        _qq_task.cancel()
        try:
            await _qq_task
        except asyncio.CancelledError:
            pass

    from app.orchestrator.planning.shadow_compare import drain_shadow_tasks
    await drain_shadow_tasks(timeout_seconds=2.0, cancel_pending=True)

    if _im_graph_bridge is not None:
        from app.agent import participle_agent as _agent_module

        _agent_module.bind_im_graph_bridge(None)
        await _im_graph_bridge.close()
        _im_graph_bridge = None
        logger.info("三 Agent Graph bridge 已关闭")

    if _conversation_health_task and not _conversation_health_task.done():
        _conversation_health_task.cancel()
        try:
            await _conversation_health_task
        except asyncio.CancelledError:
            pass

    await _conversation_service.close()
    from app.recipe_detail_store import close_recipe_detail_store
    await close_recipe_detail_store()
    logger.info("会话存储连接已关闭")


# ─── FastAPI 应用 ─────────────────────────────────────────────────────

# FastAPI 应用对象。
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

api = APIRouter(prefix=settings.API_PREFIX, redirect_slashes=False)

# 核心 HTTP API（/、/health、/chat SSE）已抽到 app/api/routes/chat.py
api.include_router(chat_router)

# 菜谱相关 API（菜谱详情查询）
api.include_router(recipes_router, prefix="/recipes", tags=["recipes"])


# ─── QQ Bot 管理 API ──────────────────────────────────────────────────

@api.get("/qqbot/status")
async def qqbot_status():
    """查看 QQ Bot 连接状态"""
    # 给前端或运维查看 QQ Bot 是否已经起来。
    if _qq_adapter is None:
        return {"enabled": False, "connected": False, "message": "QQ Bot 未启用"}
    return {
        "enabled": True,
        "connected": _qq_adapter.is_connected,
        "app_id": _qq_adapter._app_id[:4] + "****" if _qq_adapter._app_id else "",
    }


# ─── WhatsApp 管理 API ─────────────────────────────────────────────────


@api.get("/whatsapp/status")
async def whatsapp_status():
    """查看 WhatsApp 连接状态"""
    # WhatsApp 微服务状态查询。
    if _wa_adapter is None:
        return {"enabled": False, "connected": False, "message": "WhatsApp 未启用"}
    try:
        status = await _wa_adapter.get_status()
        return {
            "enabled": True,
            "connected": status.connected,
            "connectionState": status.connection_state,
            "phoneNumber": status.phone_number,
            "uptime": status.uptime,
        }
    except Exception as e:
        return {"enabled": True, "connected": False, "error": str(e)}


@api.get("/whatsapp/qr")
async def whatsapp_qr():
    """获取 WhatsApp QR 码"""
    # 获取 WhatsApp 登录二维码。
    if _wa_adapter is None:
        return {"error": "WhatsApp 未启用"}
    try:
        qr = await _wa_adapter.get_qr_code()
        return {"qr": qr}
    except Exception as e:
        return {"error": str(e)}


@api.get("/whatsapp/qr/image")
async def whatsapp_qr_image():
    """返回浏览器可直接显示和扫码的 WhatsApp 登录二维码 PNG。"""
    if _wa_adapter is None:
        return Response("WhatsApp 未启用", status_code=503, media_type="text/plain; charset=utf-8")
    try:
        status = await _wa_adapter.get_status()
        if status.connected:
            return Response("WhatsApp 已连接，无需扫码", status_code=409, media_type="text/plain; charset=utf-8")
        qr = await _wa_adapter.get_qr_code()
        if not qr:
            return Response("二维码正在生成，请稍后刷新", status_code=503, media_type="text/plain; charset=utf-8")
        import qrcode

        image = qrcode.make(qr)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    except Exception as exc:
        return Response(str(exc), status_code=500, media_type="text/plain; charset=utf-8")


@api.post("/whatsapp/login/qr/start")
async def whatsapp_login_qr_start():
    """退出当前 WhatsApp 账号并启动一轮新的扫码登录。"""
    if _wa_adapter is None:
        return Response(
            content='{"error":"WhatsApp 未启用"}',
            status_code=503,
            media_type="application/json",
        )
    try:
        return await _wa_adapter.start_qr_login()
    except Exception as exc:
        return Response(
            content=json.dumps({"error": str(exc)}, ensure_ascii=False),
            status_code=502,
            media_type="application/json",
        )


@api.get("/whatsapp/login", response_class=HTMLResponse)
async def whatsapp_login_page():
    """WhatsApp 扫码登录和账号更换页面。"""
    return HTMLResponse(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CookClaw WhatsApp 登录</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;text-align:center;padding:24px;background:#f6f7f9;color:#17212b}
.card{max-width:520px;margin:auto;padding:28px;background:#fff;border-radius:16px;box-shadow:0 8px 30px #0001}
#qr{display:none;width:min(82vw,360px);height:auto;margin:18px auto;border:12px solid #fff;box-shadow:0 2px 16px #0002}
button{border:0;border-radius:10px;padding:12px 22px;background:#16833b;color:#fff;font-size:16px;cursor:pointer}
button:disabled{opacity:.55;cursor:wait}.ok{color:#16833b;font-weight:600}.error{color:#b42318}.hint{color:#555;line-height:1.7}
.warning{margin-top:20px;padding:10px;border-radius:8px;background:#fff7e6;color:#7a4b00;font-size:13px;line-height:1.6}
</style></head>
<body><div class="card"><h2>CookClaw WhatsApp 登录</h2><p id="status">正在检查连接状态…</p>
<img id="qr" alt="WhatsApp 登录二维码"><p><button id="start" type="button">生成二维码</button></p>
<p class="hint">打开 WhatsApp → 设置 → 关联设备 → 关联设备，然后扫描二维码</p>
<p class="warning">仅限授权人员操作。“更换账号”会退出当前机器人账号，并使旧会话立即失效。</p></div>
<script>
const qr=document.getElementById('qr'), statusEl=document.getElementById('status'), startBtn=document.getElementById('start');
let connected=false, loginStarted=false, busy=false;
function setStatus(text,cls=''){statusEl.textContent=text;statusEl.className=cls}
function showQr(){qr.src='./qr/image?t='+Date.now();qr.style.display='block'}
async function jsonFetch(url,options={}){const r=await fetch(url,{cache:'no-store',...options});const data=await r.json();if(!r.ok)throw new Error(data.error||('HTTP '+r.status));return data}
async function checkStatus(){
  try{
    const s=await jsonFetch('./status');connected=Boolean(s.connected);
    if(connected){loginStarted=false;qr.style.display='none';startBtn.disabled=false;startBtn.textContent='更换账号';setStatus('✅ WhatsApp 已连接'+(s.phoneNumber?'：'+s.phoneNumber:''),'ok');return}
    startBtn.textContent='生成二维码';
    if(loginStarted){setStatus('等待扫码连接…');await refreshQr()}else{setStatus(s.error?'服务暂不可用：'+s.error:'尚未登录，请点击“生成二维码”')}
  }catch(e){setStatus('服务暂不可用：'+e.message,'error')}
}
async function refreshQr(){
  try{const data=await jsonFetch('./qr');if(data.qr){showQr();setStatus('二维码已生成，请使用 WhatsApp 扫码')}else{qr.style.display='none';setStatus('正在生成二维码，请稍候…')}}catch(e){setStatus('获取二维码失败：'+e.message,'error')}
}
async function startLogin(){
  if(busy)return;
  if(connected&&!confirm('更换账号会退出当前 WhatsApp 机器人账号，确定继续吗？'))return;
  busy=true;startBtn.disabled=true;qr.style.display='none';setStatus('正在初始化新的登录会话…');
  try{await jsonFetch('./login/qr/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});loginStarted=true;connected=false;await refreshQr()}
  catch(e){setStatus('启动扫码登录失败：'+e.message,'error')}
  finally{busy=false;startBtn.disabled=false}
}
startBtn.addEventListener('click',startLogin);checkStatus();setInterval(checkStatus,2500);
</script></body></html>""",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@api.post("/whatsapp/webhook")
async def whatsapp_webhook(
    payload: dict,
    x_webhook_secret: str | None = Header(
        default=None,
        alias="X-Webhook-Secret",
    ),
):
    """接收 WhatsApp 微服务的 Webhook 回调"""
    # WhatsApp 微服务把入站消息 POST 到这里。
    _require_webhook_secret(
        settings.WHATSAPP_WEBHOOK_SECRET,
        x_webhook_secret,
        channel="whatsapp",
    )
    if _wa_adapter is None:
        return {"error": "WhatsApp 未启用"}

    await _wa_adapter.handle_webhook(payload)
    return {"ok": True}


# ─── 微信机器人管理 API ─────────────────────────────────────────────────


@api.get("/weixin/status")
async def weixin_status():
    """查看微信机器人连接状态"""
    # 微信通道是否已启动、是否已登录。
    if _wx_adapter is None:
        return {"enabled": False, "connected": False, "message": "微信机器人未启用"}
    try:
        status = await _wx_adapter.get_status()
        return {
            "enabled": True,
            "started": status.started,
            "connected": status.connected,
            "auth_expired": status.auth_expired,
            "last_error": status.last_error,
            "token": status.token,
            "account_count": status.account_count,
            "started_count": status.started_count,
            "connected_count": status.connected_count,
            "auth_expired_count": status.auth_expired_count,
            "max_accounts": status.max_accounts,
            "accounts": status.accounts,
        }
    except Exception as e:
        return {"enabled": True, "started": False, "error": str(e)}


@api.post("/weixin/login/qr")
async def weixin_login_qr():
    """微信 QR 码登录（一步完成，最长等待 5 分钟）"""
    # 一步式登录：获取二维码并等待扫码确认。
    if _wx_adapter is None:
        return {"error": "微信机器人未启用"}
    try:
        result = await _wx_adapter.login_with_qr()
        await _handle_wx_login_result(result)   # 持久化 + set_token
        await _wx_restart_listening(result.get("accountId") or "")
        return result
    except Exception as e:
        return {"error": str(e)}


@api.post("/weixin/login/qr/start")
async def weixin_login_qr_start():
    """获取微信登录二维码（快速返回，不等待扫码确认）"""
    # 两步式登录第一步：只拿二维码，不阻塞等待。
    if _wx_adapter is None:
        return {"error": "微信机器人未启用"}
    try:
        result = await _wx_adapter.start_qr_login()
        return result
    except Exception as e:
        return {"error": str(e)}


@api.get("/weixin/login/qr/image")
async def weixin_login_qr_image(data: str = Query(..., min_length=1)):
    """把微信登录 URL 渲染成二维码 PNG，供浏览器登录页展示。"""
    try:
        import qrcode

        image = qrcode.make(data)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}


@api.post("/weixin/login/qr/wait")
async def weixin_login_qr_wait(payload: dict):
    """等待微信二维码扫码确认（长轮询，最长 5 分钟）"""
    # 两步式登录第二步：传入 session_key，等待扫码确认。
    if _wx_adapter is None:
        return {"error": "微信机器人未启用"}
    session_key = payload.get("session_key", "")
    if not session_key:
        return {"error": "session_key 必填"}
    try:
        result = await _wx_adapter.wait_qr_login(session_key)
        # 只持久化 + set_token；启动监听由页面随后的 /weixin/start（restart-safe）统一做
        return await _handle_wx_login_result(result)
    except Exception as e:
        return {"error": str(e)}


@api.post("/weixin/start")
async def weixin_start(payload: dict | None = None):
    """启动微信消息监听。扫码登录成功后，调这个接口开始收消息。"""
    if _wx_adapter is None:
        return {"error": "微信机器人未启用"}
    account_id = str((payload or {}).get("account_id") or (payload or {}).get("accountId") or "")
    try:
        await _wx_restart_listening(account_id)
        return {"ok": True, "message": "微信机器人监听已启动", "account_id": account_id}
    except Exception as e:
        from app.weixinbot.adapter import WeixinAuthExpiredError
        if isinstance(e, WeixinAuthExpiredError):
            _clear_wx_token(account_id)
            return {"error": "auth_expired", "message": "微信登录已过期，请重新扫码"}
        return {"error": str(e)}


@api.post("/weixin/accounts/remove")
async def weixin_remove_account(payload: dict):
    """停止并删除一个微信账号的本地登录态。"""
    if _wx_adapter is None:
        return {"error": "微信机器人未启用"}
    account_id = str(payload.get("account_id") or payload.get("accountId") or "")
    if not account_id:
        return {"error": "account_id 必填"}
    try:
        result = await _wx_adapter.remove_account(account_id)
        _clear_wx_token(account_id)
        return result
    except Exception as e:
        return {"error": str(e)}


@api.post("/weixin/webhook")
async def weixin_webhook(
    payload: dict,
    x_webhook_secret: str | None = Header(
        default=None,
        alias="X-Webhook-Secret",
    ),
):
    """接收微信微服务的 Webhook 回调"""
    # 微信微服务把入站消息转回 Python 主服务。
    _require_webhook_secret(
        settings.WEIXIN_WEBHOOK_SECRET,
        x_webhook_secret,
        channel="weixin",
    )
    if _wx_adapter is None:
        return {"error": "微信机器人未启用"}

    await _wx_adapter.handle_webhook(payload)
    return {"ok": True}


app.include_router(api)


@app.head("/", include_in_schema=False)
async def root_probe():
    """兼容负载均衡默认使用 ``HEAD /`` 的存活探测。"""
    return Response(status_code=200)


@app.get("/chat")
async def chat_page():
    """调试聊天页面"""
    return FileResponse("app/static/chat.html")


@app.get("/weixin/login")
async def weixin_login_page():
    """微信扫码登录页面"""
    return FileResponse("app/static/weixin-login.html")


if __name__ == "__main__":
    # 直接执行这个文件时，启动开发服务器。
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
