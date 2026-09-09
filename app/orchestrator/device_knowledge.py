"""设备产品知识与用户设备实例查询的确定性分流。"""
from __future__ import annotations

import re


_DEVICE_TERMS = (
    "田螺云厨", "cookclaw", "kitchen idea", "设备", "机器", "烹饪机",
    "料理机", "智能锅", "锅",
)
_PRODUCT_KNOWLEDGE_MARKERS = (
    "有哪些", "有什么", "型号", "哪款", "哪个好", "怎么选", "如何选",
    "区别", "对比", "介绍", "功能", "能干什么", "怎么用", "如何使用",
    "使用方法", "使用说明", "怎么绑定", "如何绑定", "多少钱", "价格",
    "购买", "where to buy", "which model", "which one", "compare",
    "features", "how to use", "how do i use", "how to bind", "price",
)
_INSTANCE_STATUS_MARKERS = (
    "在线", "离线", "空闲", "忙不忙", "运行中", "状态", "连上没",
    "连上了吗", "是否连接", "能做吗", "烹饪进度",
    "online", "offline", "idle", "busy", "status", "connected",
    "cooking progress",
)
_BOUND_INSTANCE_MARKERS = (
    "我的设备", "我绑定的", "账号下", "已绑定哪些", "绑定了哪些",
    "办公室设备", "展厅设备", "my device", "my devices", "bound devices",
)


def is_device_product_question(text: str) -> bool:
    """产品介绍/选购/用法不查询用户账号下的真实设备。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or not any(term.lower() in value for term in _DEVICE_TERMS):
        return False
    if any(marker.lower() in value for marker in _INSTANCE_STATUS_MARKERS):
        return False
    if any(marker.lower() in value for marker in _BOUND_INSTANCE_MARKERS):
        return False
    return any(marker.lower() in value for marker in _PRODUCT_KNOWLEDGE_MARKERS)


def is_device_instance_question(text: str) -> bool:
    """是否在询问账号下真实设备实例；这类问题不得交给自由回复猜状态。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or not any(term.lower() in value for term in _DEVICE_TERMS):
        return False
    return any(
        marker.lower() in value
        for marker in (*_INSTANCE_STATUS_MARKERS, *_BOUND_INSTANCE_MARKERS)
    )


def device_product_answer(text: str, lang: str = "zh") -> str:
    """只使用项目已经确认的设备能力，不编造型号、价格或购买结论。"""
    value = str(text or "").lower()
    comparison = any(marker in value for marker in (
        "有哪些", "有什么", "型号", "哪款", "哪个好", "怎么选", "区别",
        "对比", "多少钱", "价格", "购买", "which", "model", "compare", "price",
    ))
    if lang == "en":
        boundary = (
            "The public CookClaw edition includes only a local mock cooking device. "
            "It contains no verified model, price, purchasing, or hardware-compatibility data. "
            if comparison else
            "The public CookClaw edition demonstrates device workflows with a local mock only. "
        )
        return (
            boundary
            + "The supported flow is: bind a device to the account, find and select a verified "
            "device recipe, run the compatibility and online-status pre-check, choose the device "
            "when more than one is available, and explicitly confirm before cooking starts. "
            "If you want the status of your own device, ask “Which of my devices are online?”"
        )
    boundary = (
        "CookClaw 公开版只提供本地 Mock 设备，不包含已核验的型号、价格、购买或硬件兼容数据，不能据此判断哪款最好。"
        if comparison else
        "CookClaw 公开版只使用本地 Mock 演示设备工作流，不连接真实硬件。"
    )
    return (
        boundary
        + "使用流程是：先绑定设备，再搜索并选中真实设备菜谱，进行兼容性和在线状态预检；"
        "有多台设备时选择目标设备，最后明确确认后才会开始烹饪。"
        "如果你想查自己账号下的设备，请明确问“我的设备有哪些”或“哪台设备在线”。"
    )
