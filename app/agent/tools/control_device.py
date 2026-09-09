"""设备控制工具

封装现有的设备控制逻辑，提供统一的工具接口。
设备控制是关键操作，需要人工确认。
"""

from typing import Optional
from app.agent.tools import Tool, ToolResult
from app.agent.skills.recipe_operation.device_control import DeviceController


class ControlDeviceTool(Tool):
    """设备控制工具

    控制智能厨房设备（启动、停止、调整参数等）。
    这是一个关键操作，需要用户确认。
    """

    name = "control_device"
    description = "控制智能厨房设备。可以启动烹饪程序、停止设备、调整参数等。需要用户确认。"
    parameters = {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": "设备 ID"
            },
            "action": {
                "type": "string",
                "enum": ["start", "stop", "pause", "resume", "adjust"],
                "description": "操作类型"
            },
            "recipe_id": {
                "type": "string",
                "description": "菜谱 ID（启动烹饪时需要）"
            },
            "parameters": {
                "type": "object",
                "description": "额外参数（如温度、时间等）"
            }
        },
        "required": ["device_id", "action"]
    }

    def __init__(self, device_controller: DeviceController):
        self.device_controller = device_controller

    async def execute(
        self,
        device_id: str,
        action: str,
        recipe_id: Optional[str] = None,
        parameters: Optional[dict] = None
    ) -> ToolResult:
        """执行设备控制

        Args:
            device_id: 设备 ID
            action: 操作类型
            recipe_id: 菜谱 ID（启动时需要）
            parameters: 额外参数

        Returns:
            ToolResult: 执行结果（可能需要确认）
        """
        try:
            # 检查设备状态
            device_status = await self.device_controller.get_status(device_id)

            if not device_status.get("online"):
                return ToolResult.error(
                    message=f"设备 {device_id} 离线，无法控制"
                )

            # 构建确认提示
            action_desc = {
                "start": "启动烹饪程序",
                "stop": "停止设备",
                "pause": "暂停设备",
                "resume": "恢复设备",
                "adjust": "调整设备参数"
            }.get(action, action)

            confirmation_prompt = f"确定要{action_desc}吗？"
            if recipe_id:
                confirmation_prompt += f"（菜谱: {recipe_id}）"

            # 返回需要确认的结果
            return ToolResult.requires_confirmation(
                confirmation_prompt=confirmation_prompt,
                data={
                    "device_id": device_id,
                    "action": action,
                    "recipe_id": recipe_id,
                    "parameters": parameters or {}
                }
            )

        except Exception as e:
            return ToolResult.error(
                message=f"设备控制失败: {str(e)}"
            )


class ConfirmDeviceControlTool(Tool):
    """确认设备控制工具

    用户确认执行设备控制操作。
    """

    name = "confirm_device_control"
    description = "确认执行设备控制操作。在用户确认后调用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": "设备 ID"
            },
            "action": {
                "type": "string",
                "description": "操作类型"
            },
            "recipe_id": {
                "type": "string",
                "description": "菜谱 ID"
            },
            "parameters": {
                "type": "object",
                "description": "额外参数"
            },
            "confirmed": {
                "type": "boolean",
                "description": "用户是否确认"
            }
        },
        "required": ["device_id", "action", "confirmed"]
    }

    def __init__(self, device_controller: DeviceController):
        self.device_controller = device_controller

    async def execute(
        self,
        device_id: str,
        action: str,
        confirmed: bool,
        recipe_id: Optional[str] = None,
        parameters: Optional[dict] = None
    ) -> ToolResult:
        """执行确认后的设备控制

        Args:
            device_id: 设备 ID
            action: 操作类型
            confirmed: 用户是否确认
            recipe_id: 菜谱 ID
            parameters: 额外参数

        Returns:
            ToolResult: 执行结果
        """
        if not confirmed:
            return ToolResult.success(
                message="已取消设备控制操作"
            )

        try:
            # 执行设备控制
            if action == "start":
                if not recipe_id:
                    return ToolResult.error(
                        message="启动烹饪需要提供菜谱 ID"
                    )
                result = await self.device_controller.start_cooking(
                    device_id=device_id,
                    recipe_id=recipe_id,
                    parameters=parameters or {}
                )
            elif action == "stop":
                result = await self.device_controller.stop(device_id)
            elif action == "pause":
                result = await self.device_controller.pause(device_id)
            elif action == "resume":
                result = await self.device_controller.resume(device_id)
            elif action == "adjust":
                result = await self.device_controller.adjust(
                    device_id=device_id,
                    parameters=parameters or {}
                )
            else:
                return ToolResult.error(
                    message=f"未知操作: {action}"
                )

            return ToolResult.success(
                data=result,
                message=f"设备控制成功: {action}"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"设备控制失败: {str(e)}"
            )


class GetDeviceStatusTool(Tool):
    """获取设备状态工具

    查询智能厨房设备的当前状态。
    """

    name = "get_device_status"
    description = "获取智能厨房设备的当前状态，包括是否在线、当前烹饪状态等。"
    parameters = {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": "设备 ID"
            }
        },
        "required": ["device_id"]
    }

    def __init__(self, device_controller: DeviceController):
        self.device_controller = device_controller

    async def execute(self, device_id: str) -> ToolResult:
        """获取设备状态

        Args:
            device_id: 设备 ID

        Returns:
            ToolResult: 设备状态
        """
        try:
            status = await self.device_controller.get_status(device_id)

            return ToolResult.success(
                data=status,
                message=f"设备 {device_id} 状态: {status.get('status', 'unknown')}"
            )

        except Exception as e:
            return ToolResult.error(
                message=f"获取设备状态失败: {str(e)}"
            )
