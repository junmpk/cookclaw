"""会话管理器

封装现有的 ConversationService，为 LLM Agent 提供统一的会话管理接口。
"""

from typing import List, Dict, Any, Optional
from app.conversation.service import ConversationService
from app.conversation.models import ConversationTurn


class ConversationManager:
    """会话管理器

    封装 ConversationService，提供 LLM Agent 需要的接口：
    - 加载会话历史
    - 保存会话历史
    - 管理会话状态
    """

    def __init__(self, conversation_service: ConversationService):
        """初始化会话管理器

        Args:
            conversation_service: 现有的会话服务
        """
        self.service = conversation_service

    async def load_messages(
        self,
        thread_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """加载会话历史

        从 ConversationService 加载最近的消息，转换为 LLM 需要的格式。

        Args:
            thread_id: 会话 ID
            limit: 最多加载多少条消息

        Returns:
            List[Dict]: 消息列表，格式为 [{"role": "user/assistant", "content": "..."}]
        """
        try:
            # 从服务加载会话
            memory = await self.service.load_memory(thread_id)

            if not memory or not memory.turns:
                return []

            # 转换格式
            messages = []
            for turn in memory.turns[-limit:]:
                # 用户消息
                if turn.user_message:
                    messages.append({
                        "role": "user",
                        "content": turn.user_message
                    })

                # 助手消息
                if turn.assistant_message:
                    messages.append({
                        "role": "assistant",
                        "content": turn.assistant_message
                    })

            return messages

        except Exception as e:
            # 如果加载失败，返回空列表（新会话）
            return []

    async def save_messages(
        self,
        thread_id: str,
        messages: List[Dict[str, Any]]
    ) -> None:
        """保存会话历史

        将消息保存到 ConversationService。
        注意：这里只保存最后一轮对话（用户 + 助手），因为 ConversationService
        是按轮次管理的。

        Args:
            thread_id: 会话 ID
            messages: 消息列表
        """
        if not messages:
            return

        # 提取最后一轮对话（最后一条用户消息和助手消息）
        user_message = None
        assistant_message = None

        for msg in reversed(messages):
            if msg["role"] == "assistant" and not assistant_message:
                assistant_message = msg["content"]
            elif msg["role"] == "user" and not user_message:
                user_message = msg["content"]

            if user_message and assistant_message:
                break

        if user_message and assistant_message:
            try:
                # 创建新的轮次
                turn = ConversationTurn(
                    user_message=user_message,
                    assistant_message=assistant_message
                )

                # 添加到会话
                memory = await self.service.load_memory(thread_id)
                if memory:
                    memory.turns.append(turn)
                else:
                    memory = self.service.create_memory(thread_id, [turn])

                await self.service.save_memory(memory)

            except Exception as e:
                # 保存失败不应该阻断流程
                print(f"保存会话失败: {e}")

    async def clear_messages(self, thread_id: str) -> None:
        """清空会话历史

        Args:
            thread_id: 会话 ID
        """
        try:
            memory = await self.service.load_memory(thread_id)
            if memory:
                memory.turns = []
                await self.service.save_memory(memory)
        except Exception as e:
            print(f"清空会话失败: {e}")

    async def get_context(self, thread_id: str) -> Dict[str, Any]:
        """获取会话上下文

        从 ConversationService 获取用户画像、设备状态等上下文信息。

        Args:
            thread_id: 会话 ID

        Returns:
            Dict: 上下文信息
        """
        try:
            memory = await self.service.load_memory(thread_id)

            if not memory:
                return {}

            context = {}

            # 用户画像
            if memory.user_profile:
                context["user_profile"] = {
                    "name": memory.user_profile.preferred_name,
                    "preferences": memory.user_profile.preferences
                }

            # 最近搜索
            if memory.recent_searches:
                context["recent_searches"] = [
                    s.query for s in memory.recent_searches[:3]
                ]

            return context

        except Exception as e:
            return {}


class MockConversationManager:
    """Mock 会话管理器（用于测试）

    使用内存存储，用于单元测试。
    """

    def __init__(self):
        """初始化 Mock 管理器"""
        self.storage: Dict[str, List[Dict[str, Any]]] = {}

    async def load_messages(
        self,
        thread_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """加载会话历史

        Args:
            thread_id: 会话 ID
            limit: 最多加载多少条消息

        Returns:
            List[Dict]: 消息列表
        """
        messages = self.storage.get(thread_id, [])
        return messages[-limit:] if messages else []

    async def save_messages(
        self,
        thread_id: str,
        messages: List[Dict[str, Any]]
    ) -> None:
        """保存会话历史

        Args:
            thread_id: 会话 ID
            messages: 消息列表
        """
        self.storage[thread_id] = messages

    async def clear_messages(self, thread_id: str) -> None:
        """清空会话历史

        Args:
            thread_id: 会话 ID
        """
        if thread_id in self.storage:
            del self.storage[thread_id]

    async def get_context(self, thread_id: str) -> Dict[str, Any]:
        """获取会话上下文

        Args:
            thread_id: 会话 ID

        Returns:
            Dict: 上下文信息
        """
        return {}
