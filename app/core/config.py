import os
from typing import Mapping, Optional

from dotenv import load_dotenv

# 启动时加载 .env 中所有变量到 os.environ
load_dotenv()


MODEL_PROFILES = {
    "qwen": {"main": "qwen3.7-plus", "qa": "qwen-plus"},
    "deepseek": {"main": "deepseek-v4-pro", "qa": "deepseek-v4-pro"},
    "zhipu": {"main": "glm-5.2", "qa": "glm-5.2"},
}


def resolve_model_profile(
    profile: str,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """解析聊天模型档位；显式 LLM_MODEL / QA_MODEL 覆盖档位默认值。"""
    normalized = str(profile or "qwen").strip().lower()
    if normalized not in MODEL_PROFILES:
        allowed = ", ".join(MODEL_PROFILES)
        raise ValueError(f"未知 LLM_MODEL_PROFILE={profile!r}，可选：{allowed}")
    selected = dict(MODEL_PROFILES[normalized])
    values = overrides or {}
    if str(values.get("LLM_MODEL") or "").strip():
        selected["main"] = str(values["LLM_MODEL"]).strip()
    if str(values.get("QA_MODEL") or "").strip():
        selected["qa"] = str(values["QA_MODEL"]).strip()
    selected["profile"] = normalized
    return selected


def _optional_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_float_env(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _bounded_int_env(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def resolve_dashscope_native_base_url(compatible_url: str) -> str:
    """从 OpenAI 兼容地址推导同地域的 DashScope 原生 API 地址。"""
    normalized = str(compatible_url or "").strip().rstrip("/")
    suffix = "/compatible-mode/v1"
    if normalized.endswith(suffix):
        return f"{normalized[:-len(suffix)]}/api/v1"
    return "https://dashscope.aliyuncs.com/api/v1"


_MODEL_SELECTION = resolve_model_profile(
    os.getenv("LLM_MODEL_PROFILE", "qwen"),
    {"LLM_MODEL": os.getenv("LLM_MODEL", ""), "QA_MODEL": os.getenv("QA_MODEL", "")},
)


class Settings:
    """应用配置，直接从环境变量读取"""

    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "CookClaw API")
    VERSION: str = os.getenv("VERSION", "0.1.0")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")

    # 阿里云百炼 OpenAI 兼容接口与模型档位
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 联网搜索使用原生接口，才能确认搜索已执行并拿到 search_info / 来源链接。
    DASHSCOPE_NATIVE_BASE_URL: str = os.getenv(
        "DASHSCOPE_NATIVE_BASE_URL",
        resolve_dashscope_native_base_url(DASHSCOPE_BASE_URL),
    ).rstrip("/")
    LLM_MODEL_PROFILE: str = _MODEL_SELECTION["profile"]
    LLM_MODEL: str = _MODEL_SELECTION["main"]
    QA_MODEL: str = _MODEL_SELECTION["qa"]
    # 意图分类默认保持 qwen-plus，确保不同回答模型的 A/B 对比使用相同路由基线。
    INTENT_MODEL: str = os.getenv("INTENT_MODEL", "qwen-plus").strip() or "qwen-plus"
    LLM_ENABLE_THINKING: Optional[bool] = _optional_bool("LLM_ENABLE_THINKING")
    QA_ENABLE_THINKING: Optional[bool] = _optional_bool("QA_ENABLE_THINKING", False)
    INTENT_ENABLE_THINKING: Optional[bool] = _optional_bool("INTENT_ENABLE_THINKING", False)
    # Bounded Planner 默认关闭；shadow 只旁路，active 还需命中通道和用户灰度。
    TURN_PLANNER_MODE: str = (
        os.getenv("TURN_PLANNER_MODE", "off").strip().lower() or "off"
    )
    TURN_PLANNER_MODEL: str = (
        os.getenv("TURN_PLANNER_MODEL", INTENT_MODEL).strip() or INTENT_MODEL
    )
    TURN_PLANNER_TIMEOUT_SECONDS: float = _bounded_float_env(
        "TURN_PLANNER_TIMEOUT_SECONDS",
        8.0,
        minimum=0.1,
    )
    TURN_PLANNER_SHADOW_MAX_INFLIGHT: int = _bounded_int_env(
        "TURN_PLANNER_SHADOW_MAX_INFLIGHT",
        4,
        minimum=1,
    )
    TURN_PLANNER_SHADOW_JSONL_PATH: str = os.getenv(
        "TURN_PLANNER_SHADOW_JSONL_PATH",
        "",
    ).strip()
    TURN_PLANNER_ACTIVE_CHANNELS: str = os.getenv(
        "TURN_PLANNER_ACTIVE_CHANNELS",
        "qq",
    )
    TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM: str = os.getenv(
        "TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM",
        "",
    )
    TURN_PLANNER_ACTIVE_ALLOW_IDENTITIES: str = os.getenv(
        "TURN_PLANNER_ACTIVE_ALLOW_IDENTITIES",
        "",
    )
    TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT: float = max(
        0.0,
        min(
            100.0,
            _bounded_float_env(
                "TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT",
                0.0,
                minimum=0.0,
            ),
        ),
    )
    TURN_PLANNER_ACTIVE_ACTIONS: str = os.getenv(
        "TURN_PLANNER_ACTIVE_ACTIONS",
        (
            "conversation.respond,conversation.clarify,recipe.search,"
            "recipe.recommend,recipe.detail,candidate.select,"
            "candidate.compare,candidate.restore_previous,menu.plan,"
            "device.prepare,device.status,web.search"
        ),
    )
    # 受控 Deep Agent 仅参与低风险推荐表达和候选选择；默认关闭，先做 QQ 灰度。
    CONVERSATION_DEEP_AGENT_ENABLED: bool = (
        os.getenv("CONVERSATION_DEEP_AGENT_ENABLED", "false").lower() == "true"
    )
    CONVERSATION_DEEP_AGENT_TIMEOUT_SECONDS: float = float(
        os.getenv("CONVERSATION_DEEP_AGENT_TIMEOUT_SECONDS", "15")
    )
    CONVERSATION_DEEP_AGENT_CHANNELS: str = os.getenv(
        "CONVERSATION_DEEP_AGENT_CHANNELS",
        "qq",
    )
    # 主开关开启后仍需命中 QQ 用户白名单或稳定百分比分桶。默认 0%，避免误全量。
    CONVERSATION_DEEP_AGENT_QQ_ALLOW_FROM: str = os.getenv(
        "CONVERSATION_DEEP_AGENT_QQ_ALLOW_FROM",
        "",
    )
    CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT: float = max(
        0.0,
        min(
            100.0,
            float(os.getenv("CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT", "0")),
        ),
    )
    # 图片推荐成功链路默认直接使用受控 Deep Agent。它只读取视觉结构和真实
    # Milvus 候选，不拥有设备写权限；独立开关便于紧急回退到旧推荐表达模型。
    IMAGE_DEEP_AGENT_ENABLED: bool = (
        os.getenv("IMAGE_DEEP_AGENT_ENABLED", "true").lower() == "true"
    )
    # 明确用户资料记忆默认对 QQ 开启；仍保留独立开关和比例用于紧急回滚。
    CONVERSATION_PROFILE_MEMORY_ENABLED: bool = (
        os.getenv("CONVERSATION_PROFILE_MEMORY_ENABLED", "true").lower()
        == "true"
    )
    CONVERSATION_PROFILE_MEMORY_CHANNELS: str = os.getenv(
        "CONVERSATION_PROFILE_MEMORY_CHANNELS",
        "qq",
    )
    CONVERSATION_PROFILE_MEMORY_QQ_ALLOW_FROM: str = os.getenv(
        "CONVERSATION_PROFILE_MEMORY_QQ_ALLOW_FROM",
        "",
    )
    CONVERSATION_PROFILE_MEMORY_ROLLOUT_PERCENT: float = max(
        0.0,
        min(
            100.0,
            float(
                os.getenv(
                    "CONVERSATION_PROFILE_MEMORY_ROLLOUT_PERCENT",
                    "100",
                )
            ),
        ),
    )
    # 通用画像事实由独立、无工具的结构化 Extractor 提出候选；默认关闭，
    # 待 PG + Milvus 写入、删除和召回链全部启用后再按现有资料灰度放量。
    CONVERSATION_GENERAL_MEMORY_ENABLED: bool = (
        os.getenv("CONVERSATION_GENERAL_MEMORY_ENABLED", "false").lower()
        == "true"
    )
    PROFILE_FACT_EXTRACTOR_MODEL: str = (
        os.getenv("PROFILE_FACT_EXTRACTOR_MODEL", INTENT_MODEL).strip()
        or INTENT_MODEL
    )
    PROFILE_FACT_EXTRACTOR_TIMEOUT_SECONDS: float = _bounded_float_env(
        "PROFILE_FACT_EXTRACTOR_TIMEOUT_SECONDS",
        6.0,
        minimum=0.1,
    )
    # PostgreSQL 是长期事实源；Milvus 只使用独立集合做用户内语义召回。
    # MEMORY_MILVUS_URI 留空时复用 RECIPE_MILVUS_URI，启用通用记忆后只允许 Server。
    MEMORY_MILVUS_URI: str = os.getenv("MEMORY_MILVUS_URI", "").strip()
    MEMORY_MILVUS_COLLECTION: str = (
        os.getenv("MEMORY_MILVUS_COLLECTION", "cookclaw_user_memory_v1").strip()
        or "cookclaw_user_memory_v1"
    )
    MEMORY_MILVUS_VECTOR_DIM: int = _bounded_int_env(
        "MEMORY_MILVUS_VECTOR_DIM",
        1024,
        minimum=1,
    )
    MEMORY_MILVUS_TIMEOUT_SECONDS: float = _bounded_float_env(
        "MEMORY_MILVUS_TIMEOUT_SECONDS",
        10.0,
        minimum=0.1,
    )
    MEMORY_MILVUS_CONCURRENCY: int = _bounded_int_env(
        "MEMORY_MILVUS_CONCURRENCY",
        4,
        minimum=1,
    )
    MEMORY_RECALL_TOP_K: int = _bounded_int_env(
        "MEMORY_RECALL_TOP_K",
        8,
        minimum=1,
    )
    MEMORY_RECALL_MIN_SCORE: float = max(
        -1.0,
        min(
            1.0,
            _bounded_float_env("MEMORY_RECALL_MIN_SCORE", 0.35, minimum=-1.0),
        ),
    )

    # 公共联网搜索（QQ / 微信 / WhatsApp / Web 共用同一 orchestrator 路由）
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
    WEB_SEARCH_MODEL: str = os.getenv("WEB_SEARCH_MODEL", "qwen-plus").strip() or "qwen-plus"
    WEB_SEARCH_STRATEGY: str = os.getenv("WEB_SEARCH_STRATEGY", "turbo").strip() or "turbo"
    WEB_SEARCH_TIMEZONE: str = os.getenv("WEB_SEARCH_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai"
    # IM 事件当前没有可靠的用户级时区。问候只使用显式配置的业务默认时区，
    # 不根据手机号、语言或 IP 猜测用户所在地。
    GREETING_TIMEZONE: str = (
        os.getenv("GREETING_TIMEZONE", WEB_SEARCH_TIMEZONE).strip()
        or "Asia/Shanghai"
    )
    WEB_SEARCH_TIMEOUT_SECONDS: float = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "20"))
    RECIPE_WEB_SEARCH_TIMEOUT_SECONDS: float = float(
        os.getenv("RECIPE_WEB_SEARCH_TIMEOUT_SECONDS", "10")
    )
    WEB_SEARCH_CACHE_TTL_SECONDS: int = int(os.getenv("WEB_SEARCH_CACHE_TTL_SECONDS", "300"))
    WEB_SEARCH_MAX_SOURCES: int = int(os.getenv("WEB_SEARCH_MAX_SOURCES", "6"))

    # WhatsApp 微服务配置
    WHATSAPP_ENABLED: bool = os.getenv("WHATSAPP_ENABLED", "false").lower() == "true"
    WHATSAPP_SERVICE_URL: str = os.getenv("WHATSAPP_SERVICE_URL", "http://localhost:3001")
    WHATSAPP_API_TOKEN: str = os.getenv("WHATSAPP_API_TOKEN", "")
    WHATSAPP_WEBHOOK_SECRET: str = os.getenv("WHATSAPP_WEBHOOK_SECRET", "")
    WHATSAPP_PROXY_URL: str = os.getenv("WHATSAPP_PROXY_URL", "")

    # 微信机器人微服务配置
    WEIXIN_ENABLED: bool = os.getenv("WEIXIN_ENABLED", "false").lower() == "true"
    WEIXIN_SERVICE_URL: str = os.getenv("WEIXIN_SERVICE_URL", "http://localhost:3003")
    WEIXIN_TOKEN: str = os.getenv("WEIXIN_TOKEN", "")
    WEIXIN_WEBHOOK_SECRET: str = os.getenv("WEIXIN_WEBHOOK_SECRET", "")
    WEIXIN_MAX_ACCOUNTS: int = max(1, min(50, int(os.getenv("WEIXIN_MAX_ACCOUNTS", "10"))))


settings = Settings()
