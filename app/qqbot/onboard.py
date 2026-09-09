"""
QR 码扫码配置流程

通过扫描二维码绑定 QQ Bot，获取 app_id 和 client_secret。
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

from app.qqbot.constants import (
    ONBOARD_POLL_INTERVAL,
    PORTAL_HOST,
    DEFAULT_API_TIMEOUT,
)
from app.qqbot.crypto import generate_aes_key, decrypt_secret
from app.qqbot.utils import build_user_agent

logger = logging.getLogger(__name__)


# ─── 扫码状态 ──────────────────────────────────────────────────────────
class BindStatus:
    NONE = 0       # 未开始
    PENDING = 1    # 等待扫码
    COMPLETED = 2  # 扫码完成
    EXPIRED = 3    # 二维码过期


@dataclass
class BindResult:
    """扫码绑定结果"""
    app_id: str
    client_secret: str
    user_openid: str


@dataclass
class BindTask:
    """绑定任务信息"""
    task_id: str
    aes_key: str  # Base64 编码的 AES 密钥
    qr_url: str


def build_connect_url(task_id: str, portal_host: str = PORTAL_HOST) -> str:
    """
    生成扫码连接 URL。

    Args:
        task_id: 绑定任务 ID
        portal_host: Portal 域名

    Returns:
        str: 扫码 URL
    """
    return (
        f"https://{portal_host}/qqbot/openclaw/connect.html"
        f"?task_id={task_id}&_wv=2&source=hermes"
    )


async def create_bind_task(
    portal_host: str = PORTAL_HOST,
    timeout: float = DEFAULT_API_TIMEOUT,
) -> BindTask:
    """
    创建绑定任务。

    1. 本地生成 AES-256 密钥
    2. 调用 API 创建绑定任务
    3. 返回任务信息和 QR 码 URL

    Args:
        portal_host: Portal 域名
        timeout: API 超时时间

    Returns:
        BindTask: 绑定任务信息

    Raises:
        RuntimeError: 创建任务失败
    """
    # 1. 生成 AES 密钥
    aes_key = generate_aes_key()

    # 2. 调用创建 API
    url = f"https://{portal_host}/lite/create_bind_task"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": build_user_agent(),
    }
    payload = {"key": aes_key}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # 接口返回信封格式 {retcode, msg, data: {...}}，需解包内层
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    task_id = inner.get("task_id")
    if not task_id:
        raise RuntimeError(f"创建绑定任务失败: 响应中缺少 task_id, data={data}")

    qr_url = build_connect_url(task_id, portal_host)

    logger.info(f"绑定任务已创建: task_id={task_id}")

    return BindTask(
        task_id=task_id,
        aes_key=aes_key,
        qr_url=qr_url,
    )


async def poll_bind_result(
    task_id: str,
    aes_key: str,
    portal_host: str = PORTAL_HOST,
    timeout: float = DEFAULT_API_TIMEOUT,
    poll_interval: float = ONBOARD_POLL_INTERVAL,
    max_polls: int = 150,  # 最多轮询 150 次 (约 5 分钟)
) -> BindResult:
    """
    轮询扫码结果，直到扫码完成或超时。

    Args:
        task_id: 绑定任务 ID
        aes_key: Base64 编码的 AES 密钥
        portal_host: Portal 域名
        timeout: API 超时时间
        poll_interval: 轮询间隔 (秒)
        max_polls: 最大轮询次数

    Returns:
        BindResult: 扫码绑定结果

    Raises:
        TimeoutError: 轮询超时
        RuntimeError: 扫码失败或二维码过期
    """
    url = f"https://{portal_host}/lite/poll_bind_result"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": build_user_agent(),
    }
    payload = {"task_id": task_id}

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(max_polls):
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # 解包信封 {retcode, msg, data: {...}}
            inner = data.get("data") if isinstance(data.get("data"), dict) else data
            bind_status = inner.get("bind_status", BindStatus.NONE)

            if bind_status == BindStatus.COMPLETED:
                # 解密凭证
                encrypted_secret = inner.get("bot_encrypt_secret", "")
                app_id = inner.get("app_id", "")
                user_openid = inner.get("user_openid", "")

                if not encrypted_secret or not app_id:
                    raise RuntimeError(f"扫码完成但缺少必要字段: data={data}")

                client_secret = decrypt_secret(encrypted_secret, aes_key)

                logger.info(f"扫码绑定成功: app_id={app_id}, user_openid={user_openid}")

                return BindResult(
                    app_id=app_id,
                    client_secret=client_secret,
                    user_openid=user_openid,
                )

            elif bind_status == BindStatus.EXPIRED:
                raise RuntimeError("二维码已过期，请重新创建绑定任务")

            elif bind_status == BindStatus.PENDING:
                logger.debug(f"等待扫码... (第 {i + 1}/{max_polls} 次轮询)")

            # 等待下一次轮询
            await asyncio.sleep(poll_interval)

    raise TimeoutError(f"轮询超时: 已轮询 {max_polls} 次，共约 {max_polls * poll_interval:.0f} 秒")


async def run_onboard_flow(
    portal_host: str = PORTAL_HOST,
) -> BindResult:
    """
    执行完整的扫码配置流程。

    1. 创建绑定任务
    2. 打印 QR 码 URL (或生成终端 QR 码)
    3. 轮询等待扫码完成
    4. 返回绑定结果

    Args:
        portal_host: Portal 域名

    Returns:
        BindResult: 扫码绑定结果
    """
    # 1. 创建绑定任务
    task = await create_bind_task(portal_host)

    # 2. 输出 QR 码 URL
    print("\n" + "=" * 60)
    print("请使用手机 QQ 扫描以下二维码链接完成绑定:")
    print(f"\n  {task.qr_url}\n")
    print("=" * 60)

    # 尝试在终端生成 QR 码
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(task.qr_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        logger.info("未安装 qrcode 库，跳过终端 QR 码生成。请手动打开上方链接。")

    # 3. 轮询等待
    result = await poll_bind_result(
        task_id=task.task_id,
        aes_key=task.aes_key,
        portal_host=portal_host,
    )

    # 4. 输出结果
    print("\n" + "=" * 60)
    print("✅ QQ Bot 绑定成功！")
    print(f"  App ID:        {result.app_id}")
    print(f"  Client Secret: {result.client_secret[:8]}...")
    print(f"  User OpenID:   {result.user_openid}")
    print("=" * 60)
    print("\n请将以下配置添加到 .env 文件:")
    print(f"  QQ_APP_ID={result.app_id}")
    print(f"  QQ_CLIENT_SECRET={result.client_secret}")

    return result


# ─── CLI 入口：python -m app.qqbot.onboard ──────────────────────────────
if __name__ == "__main__":
    async def _cli_main():
        # 1. 建绑定任务（命中 q.qq.com/lite/create_bind_task）
        try:
            task = await create_bind_task()
        except Exception as e:
            print(f"ONBOARD_ERROR 建绑定任务失败: error_type={type(e).__name__}")
            return

        print(f"QR_URL: {task.qr_url}")

        # 2. 存成 PNG（更易扫）+ 终端 ASCII 兜底
        try:
            import qrcode
            qrcode.make(task.qr_url).save("/tmp/qqbot_bind_qr.png")
            print("QR_PNG: /tmp/qqbot_bind_qr.png")
            qr = qrcode.QRCode(border=1)
            qr.add_data(task.qr_url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception as e:
            print(f"QR_PNG_WARN 生成图片失败（不影响URL）: error_type={type(e).__name__}")

        # 3. 轮询等待扫码（最多约 5 分钟）
        print("WAITING 等待手机 QQ 扫码授权中（最多 5 分钟）...")
        try:
            result = await poll_bind_result(task.task_id, task.aes_key)
        except Exception as e:
            print(f"ONBOARD_ERROR 轮询失败: error_type={type(e).__name__}")
            return

        print(f"BIND_OK app_id={result.app_id} user_openid={result.user_openid}")
        print(f"BIND_SECRET {result.client_secret}")

    asyncio.run(_cli_main())
