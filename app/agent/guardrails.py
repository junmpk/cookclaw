"""确定性护栏

为 LLM Agent 添加安全防护和准确性验证，确保：
1. 设备控制必须经过人工确认
2. 菜谱回复必须基于 RAG 结果（禁止幻觉）
3. 关键操作有审计日志
4. 内容过滤（防止有害回复）
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """护栏检查结果

    Attributes:
        passed: 是否通过检查
        reason: 失败原因（如果失败）
        warnings: 警告信息列表
        audit_log: 审计日志条目
    """
    passed: bool
    reason: str = ""
    warnings: List[str] = None
    audit_log: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RecipeGroundingChecker:
    """菜谱真实性检查器

    确保 LLM 回复中的菜谱信息都来自 RAG 搜索结果，防止幻觉。
    """

    def __init__(self):
        # 常见的幻觉模式
        self.hallucination_patterns = [
            r"我推荐.*菜谱.*但是.*没有找到",
            r"虽然.*没有.*但是.*可以",
            r"系统里.*没有.*不过.*建议",
        ]

    def check(
        self,
        llm_response: str,
        search_results: List[Dict[str, Any]]
    ) -> GuardrailResult:
        """检查 LLM 回复是否基于真实的搜索结果

        Args:
            llm_response: LLM 生成的回复
            search_results: RAG 搜索返回的菜谱列表

        Returns:
            GuardrailResult: 检查结果
        """
        warnings = []

        # 提取搜索结果中的菜谱名称
        recipe_names = set()
        for result in search_results:
            name = result.get("name", "")
            if name:
                recipe_names.add(name)

        # 检查 LLM 回复中是否提到了不在搜索结果中的菜谱
        # 使用【】标记的菜谱名称
        mentioned_recipes = re.findall(r"【([^】]+)】", llm_response)
        hallucinated_recipes = []

        for recipe in mentioned_recipes:
            if recipe not in recipe_names:
                hallucinated_recipes.append(recipe)

        if hallucinated_recipes:
            return GuardrailResult(
                passed=False,
                reason=f"LLM 回复中提到了不存在的菜谱: {', '.join(hallucinated_recipes)}",
                warnings=[f"幻觉菜谱: {recipe}" for recipe in hallucinated_recipes]
            )

        # 检查是否有明显的幻觉模式
        for pattern in self.hallucination_patterns:
            if re.search(pattern, llm_response):
                warnings.append(f"检测到可能的幻觉模式: {pattern}")

        # 构建审计日志
        audit_log = {
            "check_type": "recipe_grounding",
            "search_result_count": len(search_results),
            "mentioned_recipes": mentioned_recipes,
            "hallucinated_recipes": hallucinated_recipes,
            "recipe_names_in_search": list(recipe_names)
        }

        return GuardrailResult(
            passed=True,
            warnings=warnings,
            audit_log=audit_log
        )


class DeviceSafetyChecker:
    """设备安全检查器

    确保设备控制操作的安全性：
    1. 必须经过人工确认
    2. 检查设备状态
    3. 防止危险操作
    """

    DANGEROUS_ACTIONS = ["start", "stop", "adjust"]

    def check(
        self,
        action: str,
        device_id: str,
        confirmed: bool,
        device_status: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """检查设备操作的安全性

        Args:
            action: 操作类型（start, stop, pause, resume, adjust）
            device_id: 设备 ID
            confirmed: 是否已经过人工确认
            device_status: 设备当前状态

        Returns:
            GuardrailResult: 检查结果
        """
        # 危险操作必须经过确认
        if action in self.DANGEROUS_ACTIONS and not confirmed:
            return GuardrailResult(
                passed=False,
                reason=f"危险操作 '{action}' 必须经过人工确认"
            )

        # 检查设备状态
        if device_status:
            if not device_status.get("online", False):
                return GuardrailResult(
                    passed=False,
                    reason=f"设备 {device_id} 离线，无法执行操作"
                )

            # 如果设备正在运行，不允许启动新任务
            if action == "start" and device_status.get("status") == "running":
                return GuardrailResult(
                    passed=False,
                    reason=f"设备 {device_id} 正在运行，无法启动新任务"
                )

        # 构建审计日志
        audit_log = {
            "check_type": "device_safety",
            "action": action,
            "device_id": device_id,
            "confirmed": confirmed,
            "device_status": device_status
        }

        return GuardrailResult(
            passed=True,
            audit_log=audit_log
        )


class ContentFilter:
    """内容过滤器

    过滤有害或不当的回复内容。
    """

    # 敏感词列表（示例，实际应用中应该更完整）
    SENSITIVE_WORDS = [
        "毒药", "自杀", "爆炸", "枪支", "毒品"
    ]

    def check(self, content: str) -> GuardrailResult:
        """检查内容是否安全

        Args:
            content: 要检查的内容

        Returns:
            GuardrailResult: 检查结果
        """
        warnings = []

        # 检查敏感词
        found_sensitive_words = []
        for word in self.SENSITIVE_WORDS:
            if word in content:
                found_sensitive_words.append(word)

        if found_sensitive_words:
            return GuardrailResult(
                passed=False,
                reason=f"内容包含敏感词: {', '.join(found_sensitive_words)}",
                warnings=[f"敏感词: {word}" for word in found_sensitive_words]
            )

        # 检查内容长度
        if len(content) > 5000:
            warnings.append("内容过长（超过 5000 字符）")

        # 构建审计日志
        audit_log = {
            "check_type": "content_filter",
            "content_length": len(content),
            "found_sensitive_words": found_sensitive_words
        }

        return GuardrailResult(
            passed=True,
            warnings=warnings,
            audit_log=audit_log
        )


class AuditLogger:
    """审计日志记录器

    记录关键操作的审计日志。
    """

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def log(
        self,
        operation: str,
        user_id: str,
        details: Dict[str, Any],
        guardrail_results: Optional[List[GuardrailResult]] = None
    ):
        """记录审计日志

        Args:
            operation: 操作类型
            user_id: 用户 ID
            details: 操作详情
            guardrail_results: 护栏检查结果
        """
        log_entry = {
            "operation": operation,
            "user_id": user_id,
            "details": details,
            "guardrail_results": [
                {
                    "passed": r.passed,
                    "reason": r.reason,
                    "warnings": r.warnings,
                    "audit_log": r.audit_log
                }
                for r in (guardrail_results or [])
            ]
        }

        self.logs.append(log_entry)
        logger.info(f"审计日志: {operation} - {user_id}")

    def get_logs(
        self,
        operation: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取审计日志

        Args:
            operation: 按操作类型过滤
            user_id: 按用户 ID 过滤

        Returns:
            List[Dict]: 日志列表
        """
        logs = self.logs

        if operation:
            logs = [log for log in logs if log["operation"] == operation]

        if user_id:
            logs = [log for log in logs if log["user_id"] == user_id]

        return logs

    def clear(self):
        """清空日志"""
        self.logs = []


# 全局实例
_recipe_grounding_checker = RecipeGroundingChecker()
_device_safety_checker = DeviceSafetyChecker()
_content_filter = ContentFilter()
_audit_logger = AuditLogger()


def check_recipe_grounding(
    llm_response: str,
    search_results: List[Dict[str, Any]]
) -> GuardrailResult:
    """检查菜谱回复的真实性

    Args:
        llm_response: LLM 生成的回复
        search_results: RAG 搜索结果

    Returns:
        GuardrailResult: 检查结果
    """
    return _recipe_grounding_checker.check(llm_response, search_results)


def check_device_safety(
    action: str,
    device_id: str,
    confirmed: bool,
    device_status: Optional[Dict[str, Any]] = None
) -> GuardrailResult:
    """检查设备操作的安全性

    Args:
        action: 操作类型
        device_id: 设备 ID
        confirmed: 是否已确认
        device_status: 设备状态

    Returns:
        GuardrailResult: 检查结果
    """
    return _device_safety_checker.check(action, device_id, confirmed, device_status)


def check_content(content: str) -> GuardrailResult:
    """检查内容安全性

    Args:
        content: 要检查的内容

    Returns:
        GuardrailResult: 检查结果
    """
    return _content_filter.check(content)


def log_audit(
    operation: str,
    user_id: str,
    details: Dict[str, Any],
    guardrail_results: Optional[List[GuardrailResult]] = None
):
    """记录审计日志

    Args:
        operation: 操作类型
        user_id: 用户 ID
        details: 操作详情
        guardrail_results: 护栏检查结果
    """
    _audit_logger.log(operation, user_id, details, guardrail_results)


def get_audit_logs(
    operation: Optional[str] = None,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """获取审计日志

    Args:
        operation: 按操作类型过滤
        user_id: 按用户 ID 过滤

    Returns:
        List[Dict]: 日志列表
    """
    return _audit_logger.get_logs(operation, user_id)
