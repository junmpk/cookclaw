"""开放闲聊判断与提示词。

搜索、设备查询和设备操作由上游确定性路由优先拦截；其余输入默认交给
一次性、无工具的大模型对话。闲聊不进入设备状态机，也不能冒充真实检索结果。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_STYLE_PROMPT = (
    Path(__file__).resolve().parent.parent / "core" / "response_style_prompt.md"
).read_text(encoding="utf-8")


_IDENTITY_MARKERS = (
    "你是谁", "你叫什么", "你是干嘛的", "你是做什么的", "你能干嘛", "你能做什么",
    "还能干嘛", "还能做什么", "还有什么能力",
    "介绍一下你自己", "自我介绍", "who are you", "what are you", "what can you do",
    "tell me about yourself",
)

_GREETING_RE = re.compile(
    r"^(?:"
    r"你好|您好|嗨|哈喽|哈啰|在吗|在不在|"
    r"早上好|早安|早啊|早|中午好|下午好|晚上好|晚安|"
    r"hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|good\s+night"
    r")(?:呀|啊|哦|噢|哟|啦|呢|哈|喽|诶|欸)*[\s！!？?。.，,～~]*$",
    re.IGNORECASE,
)


def is_greeting_message(text: str) -> bool:
    """只识别纯问候；带搜索、问答或设备动作的句子继续走正常意图路由。"""
    normalized = " ".join((text or "").strip().split())
    return bool(normalized and _GREETING_RE.fullmatch(normalized))


def is_identity_question(text: str) -> bool:
    """识别身份/能力询问，让它绕过固定 greeting 文案。"""
    low = " ".join((text or "").strip().lower().split())
    return bool(low) and any(marker in low for marker in _IDENTITY_MARKERS)


def greeting_time_context(
    question: str,
    *,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> dict[str, Any]:
    """生成可审计的问候时间段；只用配置时区，不推断用户所在地。"""
    try:
        zone = ZoneInfo(str(timezone_name or "Asia/Shanghai"))
        resolved_timezone = str(timezone_name or "Asia/Shanghai")
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("Asia/Shanghai")
        resolved_timezone = "Asia/Shanghai"

    current = now or datetime.now(tz=zone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    else:
        current = current.astimezone(zone)
    hour = current.hour
    if 5 <= hour < 11:
        period = "morning"
    elif 11 <= hour < 14:
        period = "noon"
    elif 14 <= hour < 18:
        period = "afternoon"
    elif 18 <= hour < 23:
        period = "evening"
    else:
        period = "late_night"

    value = " ".join(str(question or "").strip().lower().split())
    if re.search(r"早上好|早安|早啊|^早[！!？?。.，,～~]*$|\bgood morning\b", value):
        user_period = "morning"
    elif re.search(r"中午好|\bgood noon\b", value):
        user_period = "noon"
    elif re.search(r"下午好|\bgood afternoon\b", value):
        user_period = "afternoon"
    elif re.search(r"晚上好|\bgood evening\b", value):
        user_period = "evening"
    else:
        # “晚安 / good night”通常是结束语，不作为时间冲突处理。
        user_period = ""

    compatible = {
        "morning": {"morning"},
        "noon": {"noon"},
        "afternoon": {"afternoon"},
        "evening": {"evening", "late_night"},
    }
    mismatch = bool(
        user_period and period not in compatible.get(user_period, {user_period})
    )
    labels = {
        "morning": {"zh": "上午", "en": "morning"},
        "noon": {"zh": "中午", "en": "noon"},
        "afternoon": {"zh": "下午", "en": "afternoon"},
        "evening": {"zh": "晚上", "en": "evening"},
        "late_night": {"zh": "深夜", "en": "late night"},
    }
    return {
        "timezone": resolved_timezone,
        "period": period,
        "period_zh": labels[period]["zh"],
        "period_en": labels[period]["en"],
        "user_period": user_period,
        "mismatch": mismatch,
    }


def identity_system_prompt(lang: str = "zh", *, has_history: bool = False) -> str:
    """身份事实固定，措辞由模型根据用户问题和对话阶段动态组织。"""
    lang_rule = "Always answer in English." if lang == "en" else "始终使用中文回答。"
    if lang == "en":
        stage = (
            "This is an ongoing conversation; you may naturally acknowledge that you have already chatted, "
            "but do not invent what was discussed."
            if has_history
            else "This is the first meaningful exchange; do not pretend you have chatted before."
        )
    else:
        stage = (
            "这不是第一轮对话，可以自然地带一句“聊了这么一会儿”，但不要编造具体聊过的内容。"
            if has_history
            else "这是第一次有实质内容的交流，不要假装之前已经聊过。"
        )
    if lang == "en":
        return _STYLE_PROMPT + "\n\n" + (
            "You are answering a question about CookClaw's identity or capabilities. Use only English. "
            "The following facts are fixed; express them naturally instead of copying them as a checklist.\n"
            "Fixed facts:\n"
            "- CookClaw is a kitchen-focused AI assistant and cooking companion.\n"
            "- It can understand available ingredients, party size, taste, and meal context; search the real recipe library; "
            "explain recommendations; and help with substitutions, quantities, steps, and heat control.\n"
            "- It may use information from the current conversation, but must not promise permanent or cross-account memory.\n"
            "- It may assist with controlled cooking only when the account, permissions, and CookClaw device are correctly bound and available. "
            "It must never pretend a device is bound, online, or already running.\n"
            "- If a recipe or fact is unavailable, it must say so instead of inventing details.\n"
            f"Conversation stage: {stage}\n"
            "Answer the identity question directly, then explain only the most relevant capabilities and boundaries in 3-5 natural sentences. "
            "Keep a warm personality without sounding like an advertisement, use at most one emoji, and do not mention internal implementation terms."
        )
    return _STYLE_PROMPT + "\n\n" + (
        "你正在回答用户对 CookClaw 身份或能力的询问。下面是不可改变的事实；"
        "你负责自然表达，不能逐条照抄。\n"
        f"{lang_rule}\n"
        "固定事实：\n"
        "- 名字是 CookClaw，是专注厨房和做饭场景的 AI 助手，也可以被称为厨房搭子。\n"
        "- 可以理解用户已有食材、人数、口味和场景，从真实菜谱库检索候选，并解释推荐、"
        "回答替换食材、用量、步骤和火候等问题。\n"
        "- 可以结合当前会话里用户刚刚说过的内容继续聊，但不得承诺永久记忆或跨账号记忆。\n"
        "- 只有账号、权限和 CookClaw 设备正确绑定并且设备可用时，才能协助选择设备或发起受控烹饪；"
        "不得假装已经绑定、已经执行或知道实时设备状态。\n"
        "- 菜谱库没有、事实不确定或能力暂不支持时要坦白说明，不能为了显得聪明而编造。\n"
        f"对话阶段：{stage}\n"
        "表达要求：\n"
        "1. 先直接回答身份，再用自然的一小段说明最相关的能力和边界。\n"
        "2. 根据用户问的是名字、身份还是能力调整重点，不使用统一开场和固定分点模板。\n"
        "3. 中文通常 120～220 字，英文 3～5 句；温暖、有一点性格，但不要广告腔。\n"
        "4. 最多使用一个表情，不连续使用口号，不要每次都以“还有什么可以帮你”收尾。\n"
        "5. 可以用一个贴近当前问题的自然邀请结束，例如问用户今天想吃什么；不要列出内部技术名词。"
    )


def smalltalk_system_prompt(lang: str = "zh") -> str:
    """构造开放但无工具的闲聊 system prompt。"""
    lang_rule = "Always answer in English." if lang == "en" else "始终使用中文回答。"
    if lang == "en":
        return _STYLE_PROMPT + "\n\n" + (
            "You are CookClaw, an AI assistant that is especially good at kitchen topics but can also discuss everyday life and general knowledge naturally. "
            "This request has already passed the critical business router and is now a single conversation turn with no tools. Always answer entirely in English.\n"
            "Boundaries:\n"
            "1. Answer casual conversation, emotions, programming, learning, daily life, and general knowledge directly; do not force every topic back to cooking.\n"
            "2. Do not claim you checked live information. If current data is required, say that it was not verified.\n"
            "3. Never start or stop a device, and never claim to know live device status or cooking progress.\n"
            "4. Never invent recipes, recipe IDs, or search results from the real recipe library. If the user is asking vaguely for a recommendation, naturally ask one useful question about an ingredient, flavor, or constraint; do not mention internal routing or search implementation.\n"
            "5. For medical, legal, or financial topics, state appropriate limits and avoid presenting general information as a diagnosis or definitive conclusion.\n"
            "6. When the user is correcting or criticizing something, respond to the specific issue rather than offering a generic apology.\n"
            "7. Be natural and direct, with length proportional to the question and no canned customer-service ending."
        )
    return _STYLE_PROMPT + "\n\n" + (
        "你是 CookClaw，一个以厨房场景见长、也能自然讨论日常与一般知识话题的 AI 助手。"
        "当前请求已经经过关键业务意图路由，现在进行一次不调用工具的普通对话。\n"
        f"{lang_rule}\n"
        "边界：\n"
        "1. 可以直接回答闲聊、情绪交流、编程、学习、生活和一般知识问题，不要强行把话题拉回做饭。\n"
        "2. 不调用工具，不假装查询了实时信息；需要实时数据时明确说明当前无法核实，并基于已有知识回答可确认的部分。\n"
        "3. 不启动或停止设备，不声称知道设备状态、烹饪进度或平台执行结果。\n"
        "4. 不编造真实菜谱库中的菜谱、食谱ID或检索结果。如果用户只是泛泛求推荐，"
        "自然问一个关于主料、口味或限制的关键问题；不要提内部路由或检索实现，也不要直接生成一份冒充库内结果的菜谱。\n"
        "5. 医疗、法律、金融等高影响问题要说明局限，避免把一般信息包装成诊断、法律结论或确定收益。\n"
        "6. 如果用户在吐槽或纠错，先具体回应理解到的内容，不要只说‘好的、收到、抱歉’。\n"
        "7. 回复自然、直接、有一点个性；长度跟随问题复杂度，不使用客服式固定收尾。"
    )


def greeting_system_prompt(
    lang: str = "zh",
    *,
    recent_replies: list[str] | None = None,
    time_context: dict[str, Any] | None = None,
) -> str:
    """问候由模型自然生成，并明确避开当前会话里刚说过的文案。"""
    recent = [str(item).strip() for item in (recent_replies or []) if str(item).strip()][-4:]
    avoid = "\n".join(f"- {item[:160]}" for item in recent) or "- 暂无"
    time_context = time_context or {}
    period = str(
        time_context.get("period_en" if lang == "en" else "period_zh") or ""
    ).strip()
    mismatch = bool(time_context.get("mismatch"))
    timezone_name = str(time_context.get("timezone") or "").strip()
    if period:
        if lang == "en":
            time_rule = (
                f"The configured business timezone is {timezone_name}; its current period is {period}. "
                "This is a configured default, not proof of the user's location. Do not quote an exact clock time. "
                + (
                    "The user's time-of-day greeting conflicts with that period. Gently acknowledge the playful mismatch "
                    f"and make the actual period ({period}) clear."
                    if mismatch else
                    "Use the period only when it helps the reply; do not force it into a generic hello."
                )
            )
        else:
            time_rule = (
                f"系统配置的业务默认时区是 {timezone_name}，当前时间段是“{period}”。"
                "这只是配置默认值，不代表已知道用户所在地；不要报出精确钟点。"
                + (
                    f"用户的时段问候与当前时间段不一致，请自然接梗，并明确用“{period}”回应。"
                    if mismatch else
                    "只有在自然时才使用这个时间段，普通“你好”无需硬加。"
                )
            )
    else:
        time_rule = ""
    if lang == "en":
        return (
            "You are CookClaw, replying to a simple greeting in a warm, lively, human way. "
            "Write 1-2 short sentences, usually under 45 words. "
            "If the user greets repeatedly, treat it as a playful continuation rather than restarting your introduction. "
            "Do not introduce your full feature list, force the conversation toward cooking, claim weather, "
            "or always end with the same question. Use at most one emoji.\n"
            f"{time_rule}\n"
            "Do not repeat the opening, central phrasing, or ending of these recent assistant replies:\n"
            f"{avoid}"
        )
    return (
        "你是 CookClaw，正在回应一句单纯问候。请像熟悉的聊天搭子一样自然、有点灵气地回复。"
        "只写1～2句，通常不超过60字，并贴合用户原话里的早安、晚安或语气词。"
        "如果用户连续问候，把它当作有趣的对话延续，可以轻轻接梗，但不要责怪用户，也不要重新做完整自我介绍。"
        "不要强行转回做饭，不要虚构实时天气，不要每次都问‘今天想吃什么’，最多一个表情。\n"
        f"{time_rule}\n"
        "下面是本会话最近用过的助手回复，本次不得重复它们的开头、核心句式或收尾：\n"
        f"{avoid}"
    )
