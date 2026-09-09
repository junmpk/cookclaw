"""
确定性执行 —— 收敛设备做菜为「受控子进程调用」，不经 Agent 自由 shell（守红线，见架构评审 #2）。

由 orchestrator 用确切的 cookId 调用公开版的本地 Mock 脚本，先检查状态，
再执行 start/stop 并复查结果。Mock 不连接真实硬件或厂商云。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from app.domain.device_execution import (
    new_device_action_identity,
    normalize_device_msg_id,
)
from app.observability.trace import trace_async_tool

logger = logging.getLogger(__name__)

_SCRIPTS = Path(__file__).resolve().parent.parent / "agent" / "skills" / "recipe-operation" / "scripts"
_EXECUTE_RECIPE = _SCRIPTS / "execute_recipe.py"
_SEND_COMMAND = _SCRIPTS / "send_device_command.py"
_DEVICE_CHECK = _SCRIPTS / "device_check.py"
_FETCH_RECIPE = _SCRIPTS / "fetch_recipe.py"


def _log_fingerprint(value: object) -> str:
    raw = str(value or "")
    if not raw:
        return "-"
    return "sha256:" + hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _recipe_ai_completion_enabled() -> bool:
    from app.orchestrator.recipe_completion import recipe_ai_completion_enabled

    return recipe_ai_completion_enabled()


_DEVICE_TEXT = {
    "zh": {
        "offline": "离线",
        "online_idle": "在线，空闲",
        "online": "在线",
        "online_unknown": "在线，但忙闲状态未知",
        "running": "在线，运行中(status={status})",
        "status_query_failed": "设备状态查询失败，请稍后再试",
        "status_parse_failed": "设备状态响应解析失败",
        "status_timeout": "设备状态查询超时",
        "config_missing": "设备配置缺失，请联系管理员配置设备后重试",
        "device_offline": "设备离线",
        "device_busy": "设备正忙（正在做另一道）",
        "command_failed": "下发失败",
        "device_not_ready": "设备未就绪",
        "device_unknown": "没有找到所选设备，请重新选择一台已配置的设备",
        "push_not_running": "启动请求已经发送，但暂时没有确认设备进入运行状态。先不要重复启动，请检查设备面板或回复“查看状态”。",
        "device_timeout": "设备响应超时",
        "command_outcome_unknown": "设备指令等待超时，暂时无法确认云端是否已接收。请不要重复启动，先回复“查看状态”。",
        "execute_exception": "执行异常，请稍后再试",
        "stop_no_ack": "设备未应答 200",
    },
    "en": {
        "offline": "offline",
        "online_idle": "online, idle",
        "online": "online",
        "online_unknown": "online, but idle/busy state is unknown",
        "running": "online, running (status={status})",
        "status_query_failed": "Device status query failed. Please try again later.",
        "status_parse_failed": "Could not parse the device status response.",
        "status_timeout": "Device status query timed out.",
        "config_missing": "Device configuration is missing. Please ask an administrator to configure it and try again.",
        "device_offline": "The device is offline.",
        "device_busy": "The device is already cooking another recipe.",
        "command_failed": "The start command failed.",
        "device_not_ready": "The device is not ready.",
        "device_unknown": "The selected device was not found. Please choose a configured device.",
        "push_not_running": "The start request was sent, but the device has not yet been confirmed as running. Do not send another start request; check the device panel or ask for status.",
        "device_timeout": "The device response timed out.",
        "command_outcome_unknown": "The device request timed out and I cannot confirm whether the cloud accepted it. Do not start it again; ask for device status first.",
        "execute_exception": "The cooking command failed unexpectedly. Please try again later.",
        "stop_no_ack": "The device did not acknowledge the stop command.",
    },
}


def _device_text(lang: str, key: str, **kwargs) -> str:
    lang = "en" if lang == "en" else "zh"
    return _DEVICE_TEXT[lang][key].format(**kwargs)


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in str(text or ""))


def list_devices() -> list:
    """[(device_id, name), ...] —— 给聊天层列设备让用户选。读 recipe-operation 的 config。

    device_config 在 scripts 目录（含连字符、非包），用 sys.path 注入后 import（同 route-A 范式）。
    失败返回 []（聊天层据此退回"单设备/默认"行为）。
    """
    try:
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        import device_config
        return device_config.list_devices(device_config.load_config())
    except Exception as e:
        logger.warning("list_devices 读取失败 error_type=%s", type(e).__name__)
        return []


def _online_state(value) -> bool | None:
    """把云端可能返回的 0/1、字符串和布尔值统一成三态。"""
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
    return None


def _is_idle_status(value) -> bool:
    """设备忙闲状态必须是明确的数值 0 或字符串 "0"，布尔值不算状态码。"""
    return not isinstance(value, bool) and value in (0, "0")


def _validate_status_payload(payload: dict) -> dict:
    """返回已验证的 data；接口失败/字段缺失不能降级成“设备离线”。"""
    if not isinstance(payload, dict):
        raise ValueError("payload_not_object")
    code = payload.get("code")
    success = payload.get("success")
    if code not in (None, 200, "200"):
        raise ValueError(f"api_code_{code}")
    if success in (False, 0, "false", "False"):
        raise ValueError("api_success_false")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("missing_data")
    if _online_state(data.get("isOnline")) is None:
        raise ValueError("invalid_online_state")
    return data


def _summarize_status(data: dict, lang: str = "zh") -> str:
    is_online = _online_state(data.get("isOnline"))
    attrs = data.get("attributes") or {}
    status = attrs.get("status")

    if is_online is not True:
        return _device_text(lang, "offline")
    if _is_idle_status(status):
        return _device_text(lang, "online_idle")
    if status in (None, ""):
        return _device_text(lang, "online_unknown")
    return _device_text(lang, "running", status=status)


def evaluate_device_readiness(status_result: dict, device_id: str = None, lang: str = "zh") -> dict:
    """把状态查询结果收敛为执行门禁：只有明确在线且 status=0 才允许下发。

    未返回忙闲状态、查询失败或找不到所选设备都按“状态未知”阻断，不能把
    缺失字段乐观地当成空闲。
    """
    devices = status_result.get("devices") or []
    selected = None
    if device_id is not None:
        selected = next(
            (
                item for item in devices
                if str(item.get("device_id") or "") == str(device_id)
            ),
            None,
        )
    elif len(devices) == 1:
        selected = devices[0]

    if selected is None:
        return {
            "ready": False,
            "code": status_result.get("code") or "DEVICE_STATUS_UNKNOWN",
            "state": "unknown",
            "device_id": device_id,
            "name": device_id or ("Device" if lang == "en" else "设备"),
            "status": "unknown" if lang == "en" else "状态未知",
            "reason": status_result.get("reason") or _device_text(lang, "status_query_failed"),
        }
    if not selected.get("ok"):
        return {
            "ready": False,
            "code": selected.get("code") or "DEVICE_STATUS_FAILED",
            "state": "unknown",
            "device_id": selected.get("device_id"),
            "name": selected.get("name") or selected.get("device_id"),
            "status": selected.get("status") or ("unknown" if lang == "en" else "状态未知"),
            "reason": selected.get("reason") or _device_text(lang, "status_query_failed"),
        }

    data = selected.get("data") or {}
    online = _online_state(data.get("isOnline"))
    status = (data.get("attributes") or {}).get("status")
    common = {
        "device_id": selected.get("device_id"),
        "name": selected.get("name") or selected.get("device_id"),
        "status": selected.get("status") or _summarize_status(data, lang),
    }
    if online is False:
        return {
            **common,
            "ready": False,
            "code": "DEVICE_OFFLINE",
            "state": "offline",
            "reason": _device_text(lang, "device_offline"),
        }
    if online is not True or status in (None, ""):
        return {
            **common,
            "ready": False,
            "code": "DEVICE_STATUS_UNKNOWN",
            "state": "unknown",
            "reason": _device_text(lang, "status_query_failed"),
        }
    if not _is_idle_status(status):
        return {
            **common,
            "ready": False,
            "code": "DEVICE_BUSY",
            "state": "busy",
            "reason": _device_text(lang, "device_busy"),
        }
    return {
        **common,
        "ready": True,
        "code": "OK",
        "state": "idle",
        "reason": "",
    }


def _is_device_running_data(data: dict) -> bool:
    attrs = data.get("attributes") or {}
    status = attrs.get("status", 0)
    return status not in (0, "0", None, "")


def _public_device_error(text: str, lang: str = "zh") -> str:
    """把脚本异常输出压缩成可给用户看的原因，避免 traceback/路径泄漏。"""
    text = (text or "").strip()
    is_en = lang == "en"
    generic = _device_text(lang, "status_query_failed")
    config_missing = _device_text(lang, "config_missing")

    if not text:
        return generic
    if "config.json" in text and (
        "不存在" in text or "No such file" in text or "FileNotFoundError" in text
    ):
        return config_missing
    if "Traceback" in text or "File \"" in text:
        logger.warning("设备脚本异常输出已脱敏 output_chars=%s", len(text))
        return generic

    for marker, key in (
        ("设备离线", "device_offline"),
        ("正在运行", "device_busy"),
        ("执行失败", "command_failed"),
        ("设备未就绪", "device_not_ready"),
        ("未知设备", "device_unknown"),
        ("unknown device", "device_unknown"),
    ):
        if marker.lower() in text.lower():
            return _device_text(lang, key)
    if len(text) > 240:
        logger.warning("设备脚本长错误输出已脱敏 output_chars=%s", len(text))
        return generic
    if is_en and _has_cjk(text):
        logger.warning("设备脚本中文错误输出已脱敏 output_chars=%s", len(text))
        return generic
    # 未识别的脚本输出可能包含内部路径、接口正文或标识符，不直接回显给客户。
    return generic


_AMBIGUOUS_DEVICE_DISPATCH_MARKERS = (
    "http 请求超时",
    "curl 错误",
    "http 响应缺少有效状态码",
    "响应不是合法 json",
    "timed out",
    "timeout",
    "connection reset",
    "empty reply",
    "unexpected eof",
)


def _device_dispatch_outcome_unknown(text: str) -> bool:
    """判断 POST 可能已被远端接收但本地没有拿到可信结果的失败。

    子进程在设备状态预检失败时同样可能出现网络错误，因此只有输出明确进入
    ``send_run_command`` 后才按未知结果处理。未知结果绝不能被自动重试。
    """
    normalized = str(text or "").lower()
    dispatch_started = "发送指令失败" in normalized or "发送设备指令" in normalized
    if not dispatch_started:
        return False
    if any(marker in normalized for marker in _AMBIGUOUS_DEVICE_DISPATCH_MARKERS):
        return True
    match = re.search(r"http 请求失败:\s*status=(\d{3})", normalized)
    return bool(match and int(match.group(1)) >= 500)


async def _terminate_subprocess(proc, *, operation: str) -> None:
    """超时后回收子进程；不让脚本在上层已返回失败后继续发送设备命令。"""
    if proc is None or getattr(proc, "returncode", None) is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    except Exception as exc:
        logger.warning(
            "%s 子进程终止失败 error_type=%s",
            operation,
            type(exc).__name__,
        )
        return
    wait = getattr(proc, "wait", None)
    if wait is None:
        return
    try:
        await asyncio.wait_for(wait(), timeout=5)
    except Exception as exc:
        logger.warning(
            "%s 子进程回收失败 error_type=%s",
            operation,
            type(exc).__name__,
        )


async def _check_one_device(lang: str, device_id: str = None, timeout: int = 30) -> dict:
    args = ["python3", str(_DEVICE_CHECK), lang]
    if device_id:
        args.append(device_id)
    started_at = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _terminate_subprocess(proc, operation="device_status")
        logger.warning(
            "设备状态查询超时 device=%s timeout=%ss",
            _log_fingerprint(device_id or "default"), timeout,
        )
        return {
            "ok": False,
            "code": "DEVICE_STATUS_TIMEOUT",
            "reason": _device_text(lang, "status_timeout"),
            "raw": "",
        }

    stdout = out.decode(errors="replace")
    stderr = err.decode(errors="replace")
    text = stdout + stderr
    elapsed_ms = round((time.monotonic() - started_at) * 1000)

    if proc.returncode != 0:
        logger.warning(
            "设备状态查询失败 device=%s rc=%s elapsed_ms=%s output_chars=%s",
            _log_fingerprint(device_id or "default"), proc.returncode, elapsed_ms, len(text),
        )
        return {"ok": False, "code": "DEVICE_STATUS_FAILED", "reason": _public_device_error(text, lang), "raw": text}

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning(
            "设备状态解析失败 device=%s elapsed_ms=%s",
            _log_fingerprint(device_id or "default"), elapsed_ms,
        )
        return {"ok": False, "code": "DEVICE_STATUS_INVALID", "reason": _device_text(lang, "status_parse_failed"), "raw": text}

    try:
        data = _validate_status_payload(payload)
    except ValueError as exc:
        logger.warning(
            "设备状态响应无效 device=%s reason=%s code=%s success=%s trace=%s elapsed_ms=%s",
            _log_fingerprint(device_id or "default"),
            exc,
            payload.get("code") if isinstance(payload, dict) else None,
            payload.get("success") if isinstance(payload, dict) else None,
            _log_fingerprint(payload.get("traceId") if isinstance(payload, dict) else None),
            elapsed_ms,
        )
        return {
            "ok": False,
            "code": "DEVICE_STATUS_FAILED",
            "reason": _device_text(lang, "status_query_failed"),
            "raw": text,
        }

    logger.info(
        "设备状态 device=%s online=%s code=%s trace=%s elapsed_ms=%s",
        _log_fingerprint(device_id or "default"),
        _online_state(data.get("isOnline")),
        payload.get("code"),
        _log_fingerprint(payload.get("traceId")),
        elapsed_ms,
    )
    return {
        "ok": True,
        "code": "OK",
        "device_id": device_id,
        "status": _summarize_status(data, lang),
        "data": data,
        "raw": text,
    }


@trace_async_tool("device_status")
async def check_device_status(lang: str = "zh", timeout: int = 30, device_id: str = None) -> dict:
    """查询设备状态；不传 device_id 时，多设备会全部查询。"""
    try:
        devices = list_devices()
        if device_id:
            configured_name = next(
                (
                    name for configured_id, name in devices
                    if str(configured_id) == str(device_id)
                ),
                device_id,
            )
            targets = [(device_id, configured_name)]
        elif len(devices) > 1:
            targets = devices
        else:
            targets = [(devices[0][0], devices[0][1])] if devices else [(None, "默认设备")]

        # 多设备状态查询相互独立，并行执行，避免某台离线复查拖慢全部结果。
        results = await asyncio.gather(*(
            _check_one_device(lang, did, timeout=timeout) for did, _ in targets
        ))
        for item, (did, name) in zip(results, targets):
            item["device_id"] = did
            item["name"] = name

        ok = all(item.get("ok") for item in results)
        return {
            "ok": ok,
            "code": "OK" if ok else next((item.get("code") for item in results if not item.get("ok")), "DEVICE_STATUS_FAILED"),
            "devices": results,
            "raw": "\n".join(item.get("raw", "") for item in results),
        }
    except asyncio.TimeoutError:
        return {"ok": False, "code": "DEVICE_STATUS_TIMEOUT", "reason": _device_text(lang, "status_timeout"), "devices": [], "raw": ""}
    except Exception as e:
        logger.error("check_device_status 异常：error_type=%s", type(e).__name__)
        return {"ok": False, "code": "DEVICE_STATUS_FAILED", "reason": _public_device_error(str(e), lang), "devices": [], "raw": ""}


def _valid_cook_id(cook_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(cook_id or "").strip()))


def _explicit_false(value) -> bool:
    return value in (False, 0, "0", "false", "False", "no", "No")


def _definitive_recipe_rejection(result: dict) -> bool:
    """只有明确不存在/不可执行才阻断；网络和详情服务故障不能伪装成不存在。"""
    return result.get("code") in {
        "MISSING_COOK_ID", "RECIPE_NOT_FOUND", "RECIPE_NOT_EXECUTABLE",
    }


def compatible_devices_for_recipe(recipe: dict, devices: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """仅在详情提供明确兼容字段时过滤；字段缺失时不凭空判定不兼容。"""
    if not isinstance(recipe, dict):
        return []
    supported_ids = recipe.get("supportedDeviceIds") or recipe.get("supported_device_ids")
    supported_types = recipe.get("supportedCookTypes") or recipe.get("supported_cook_types")
    id_set = {str(item) for item in supported_ids or []} if isinstance(supported_ids, list) else set()
    type_set = {str(item) for item in supported_types or []} if isinstance(supported_types, list) else set()
    if not id_set and not type_set:
        return list(devices or [])
    try:
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        import device_config
        config = device_config.load_config()
    except Exception:
        # 无法读取设备能力不是“不兼容”；让后续设备状态/命令返回真实原因。
        return list(devices or [])
    compatible = []
    for device_id, name in devices or []:
        try:
            dev = device_config.resolve_device(config, device_id)
        except Exception:
            continue
        if id_set and str(device_id) not in id_set and str(dev.get("provider_device_id") or "") not in id_set:
            continue
        if type_set and str(dev.get("device_type")) not in type_set:
            continue
        compatible.append((device_id, name))
    return compatible


@trace_async_tool("recipe_detail")
async def fetch_recipe_details(
    cook_id: str,
    lang: str = "zh",
    timeout: int = 25,
    *,
    prefer_cache: bool = False,
) -> dict:
    """按真实 cookId 获取远程详情；供步骤展示和启动前存在性校验共用。"""
    cook_id = str(cook_id or "").strip()
    if not _valid_cook_id(cook_id):
        reason = (
            "The recipe is missing a valid execution ID."
            if lang == "en" else "菜谱缺少有效的执行 ID。"
        )
        return {"ok": False, "code": "MISSING_COOK_ID", "reason": reason, "recipe": {}}

    if prefer_cache:
        try:
            from app.recipe_detail_store import load_recipe_detail

            cached = await load_recipe_detail(
                cook_id,
                lang,
                max_age_seconds=max(
                    0,
                    int(os.getenv("RECIPE_DETAIL_CACHE_TTL_SECONDS", "86400")),
                ),
            )
            if cached:
                cached_detail = cached["detail"]
                if _recipe_ai_completion_enabled():
                    from app.orchestrator.recipe_completion import (
                        complete_missing_recipe_fields,
                    )

                    completed_detail = await complete_missing_recipe_fields(
                        cached_detail,
                    )
                    if completed_detail is not cached_detail:
                        from app.recipe_detail_store import save_recipe_detail

                        await save_recipe_detail(
                            completed_detail,
                            cached.get("raw_payload") or {},
                        )
                        cached_detail = completed_detail
                return {
                    "ok": True,
                    "code": "OK",
                    "reason": "",
                    "recipe": cached_detail,
                    "source": "postgres_cache",
                    "trace_id": None,
                }
        except Exception as exc:
            logger.warning(
                "读取菜谱详情缓存失败，不阻断远端查询 cook_id=%s error_type=%s",
                _log_fingerprint(cook_id),
                type(exc).__name__,
            )

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_FETCH_RECIPE),
            cook_id,
            lang,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout = out.decode(errors="replace")
        stderr = err.decode(errors="replace")
        if proc.returncode != 0:
            logger.warning(
                "菜谱详情查询失败 cook_id_hash=%s rc=%s stderr_chars=%s",
                hashlib.sha256(cook_id.encode()).hexdigest()[:10], proc.returncode, len(stderr),
            )
            reason = (
                "The recipe detail service could not verify this recipe, so no device command was sent."
                if lang == "en" else "详情服务没有核验到这条菜谱，所以我没有向设备发送命令。"
            )
            return {"ok": False, "code": "RECIPE_LOOKUP_FAILED", "reason": reason, "recipe": {}}
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "code": "RECIPE_RESPONSE_INVALID",
                "reason": "The recipe detail response was invalid." if lang == "en" else "菜谱详情响应格式无效。",
                "recipe": {},
            }
        api_code = payload.get("code") if isinstance(payload, dict) else None
        success = payload.get("success") if isinstance(payload, dict) else None
        if api_code not in (None, 200, "200") or success in (False, 0, "false", "False"):
            not_found = api_code in (404, "404")
            code = "RECIPE_NOT_FOUND" if not_found else "RECIPE_LOOKUP_FAILED"
            if not_found:
                reason = (
                    "This recipe was not found in the execution service, so no device command was sent."
                    if lang == "en" else "设备执行服务明确返回这条菜谱不存在，所以我没有发送启动命令。"
                )
            else:
                reason = (
                    "The recipe detail service is temporarily unavailable. The recipe was not marked as missing."
                    if lang == "en" else "菜谱详情服务暂时不可用，但这不代表菜谱不存在。"
                )
            return {"ok": False, "code": code, "reason": reason, "recipe": {}}
        recipe = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(recipe, dict) or not recipe:
            return {
                "ok": False,
                "code": "RECIPE_NOT_FOUND",
                "reason": "This recipe has no executable record." if lang == "en" else "这条菜谱没有可执行记录。",
                "recipe": {},
            }
        if any(_explicit_false(recipe.get(key)) for key in (
            "executable", "isExecutable", "canExecute", "deviceExecutable", "supportsDevice",
        ) if key in recipe):
            return {
                "ok": False,
                "code": "RECIPE_NOT_EXECUTABLE",
                "reason": (
                    "This recipe record is not marked as device-executable, so no start command was sent."
                    if lang == "en" else "这条菜谱记录明确标记为不可由设备执行，所以我没有发送启动命令。"
                ),
                "recipe": recipe,
            }
        try:
            from app.orchestrator.recipe_detail import (
                normalize_recipe_detail,
                sanitize_recipe_detail_payload,
            )

            normalized_recipe = normalize_recipe_detail(recipe, language=lang)
            if _recipe_ai_completion_enabled():
                from app.orchestrator.recipe_completion import (
                    complete_missing_recipe_fields,
                )

                normalized_recipe = await complete_missing_recipe_fields(
                    normalized_recipe,
                )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "菜谱详情规范化失败 cook_id=%s error_type=%s",
                _log_fingerprint(cook_id),
                type(exc).__name__,
            )
            return {
                "ok": False,
                "code": "RECIPE_RESPONSE_INVALID",
                "reason": (
                    "The recipe detail response could not be normalized."
                    if lang == "en" else "菜谱详情响应无法转换为可展示格式。"
                ),
                "recipe": {},
            }
        try:
            from app.recipe_detail_store import save_recipe_detail

            await save_recipe_detail(
                normalized_recipe,
                sanitize_recipe_detail_payload(recipe),
            )
        except Exception as exc:
            # PG 暂时不可用不能把远端真实详情降级成失败；本轮仍直接展示。
            logger.warning(
                "保存菜谱详情失败，不阻断本轮展示 cook_id=%s error_type=%s",
                _log_fingerprint(cook_id),
                type(exc).__name__,
            )
        return {
            "ok": True,
            "code": "OK",
            "reason": "",
            "recipe": normalized_recipe,
            "source": "remote",
            "trace_id": payload.get("traceId"),
        }
    except asyncio.TimeoutError:
        await _terminate_subprocess(proc, operation="recipe_detail")
        return {
            "ok": False,
            "code": "RECIPE_LOOKUP_TIMEOUT",
            "reason": "Recipe verification timed out; no device command was sent." if lang == "en" else "菜谱核验超时，本次没有向设备发送命令。",
            "recipe": {},
        }
    except Exception as exc:
        logger.error("菜谱详情核验异常: error_type=%s", type(exc).__name__)
        return {
            "ok": False,
            "code": "RECIPE_LOOKUP_FAILED",
            "reason": "Recipe verification failed; no device command was sent." if lang == "en" else "菜谱核验失败，本次没有向设备发送命令。",
            "recipe": {},
        }


@trace_async_tool("device_start")
async def execute_cook(
    cook_id: str,
    lang: str = "zh",
    timeout: int = 45,
    device_id: str = None,
    *,
    msg_id: int | None = None,
) -> dict:
    """确定性下发做菜：execute_recipe.py <cookId> <lang> [device_id] [msg_id]。

    device_id 透传给脚本（不传=默认设备）；检查与下发用同一台，避免查 A 发 B。
    返回 {"ok": bool, "reason": str, "raw": str}。
    reason 仅在失败时给（设备离线 / 设备忙 / 执行失败 / 超时 / 异常）。
    """
    cook_id = str(cook_id or "").strip()
    if not _valid_cook_id(cook_id):
        return {
            "ok": False,
            "code": "MISSING_COOK_ID",
            "reason": "The recipe is missing a valid execution ID." if lang == "en" else "菜谱缺少有效的执行 ID。",
            "command_sent": False,
            "started": False,
            "retryable": False,
            "raw": "",
        }
    try:
        command_msg_id = normalize_device_msg_id(
            msg_id if msg_id is not None else new_device_action_identity()[1]
        )
    except ValueError:
        return {
            "ok": False,
            "code": "INVALID_DEVICE_MSG_ID",
            "reason": "The device request ID is invalid." if lang == "en" else "设备请求 ID 无效。",
            "command_sent": False,
            "started": False,
            "retryable": False,
            "raw": "",
        }

    # 用户最终确认后的实时门禁。只有所选设备明确“在线且空闲”才继续；
    # execute_recipe.py 内仍会在真正下发前复查一次，防止检查后的状态竞态。
    status_check = await check_device_status(
        lang=lang,
        timeout=min(20, timeout),
        device_id=device_id,
    )
    readiness = evaluate_device_readiness(status_check, device_id=device_id, lang=lang)
    if not readiness.get("ready"):
        logger.info(
            "execute_cook 状态门禁阻断 device=%s state=%s code=%s",
            _log_fingerprint(device_id or "default"),
            readiness.get("state"),
            readiness.get("code"),
        )
        return {
            "ok": False,
            "code": readiness.get("code") or "DEVICE_STATUS_UNKNOWN",
            "reason": readiness.get("reason") or _device_text(lang, "device_not_ready"),
            "command_sent": False,
            "started": False,
            "retryable": True,
            "readiness": readiness,
            "msg_id": command_msg_id,
            "raw": status_check.get("raw") or "",
        }
    proc = None
    try:
        args = ["python3", str(_EXECUTE_RECIPE), str(cook_id), lang]
        args.extend([device_id or "", str(command_msg_id)])
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        text = out.decode(errors="replace") + err.decode(errors="replace")

        pushed = proc.returncode == 0 and "已开始制作" in text
        reason = ""
        if not pushed:
            if _device_dispatch_outcome_unknown(text):
                logger.warning(
                    "execute_cook 下发结果未知 cook_id=%s device=%s rc=%s output_chars=%s",
                    _log_fingerprint(cook_id),
                    _log_fingerprint(device_id or "default"),
                    proc.returncode,
                    len(text),
                )
                return {
                    "ok": False,
                    "code": "DEVICE_OUTCOME_UNKNOWN",
                    "reason": _device_text(lang, "command_outcome_unknown"),
                    "command_sent": None,
                    "started": None,
                    "retryable": False,
                    "msg_id": command_msg_id,
                    "raw": text,
                }
            for marker, key in (("设备离线", "device_offline"), ("正在运行", "device_busy"),
                                ("执行失败", "command_failed")):
                if marker in text:
                    reason = _device_text(lang, key)
                    break
            reason = reason or _public_device_error(text, lang)
            logger.warning(
                "execute_cook 失败 cook_id=%s device=%s rc=%s output_chars=%s",
                _log_fingerprint(cook_id), _log_fingerprint(device_id or "default"),
                proc.returncode, len(text),
            )
            code = "DEVICE_COMMAND_REJECTED"
            if reason == _device_text(lang, "device_offline"):
                code = "DEVICE_OFFLINE"
            elif reason == _device_text(lang, "device_busy"):
                code = "DEVICE_BUSY"
            elif any(marker in text.lower() for marker in ("不支持", "不兼容", "unsupported", "incompatible")):
                code = "DEVICE_INCOMPATIBLE"
            elif any(marker in text.lower() for marker in ("食谱不存在", "recipe not found", "cookid不存在")):
                code = "RECIPE_NOT_FOUND"
            return {
                "ok": False, "code": code, "reason": reason,
                "command_sent": False, "started": False, "retryable": True,
                "msg_id": command_msg_id, "raw": text,
            }

        # 平台返回 200 只代表消息被接收，不代表设备真的开始运行。
        # 短轮询确认同一台设备进入运行态，避免 QQ 误报"设备运转中"。
        verify_raw = []
        for _ in range(3):
            await asyncio.sleep(2)
            status = await _check_one_device(lang, device_id, timeout=15)
            verify_raw.append(status.get("raw", ""))
            if status.get("ok") and _is_device_running_data(status.get("data") or {}):
                return {
                    "ok": True, "code": "OK", "reason": "",
                    "command_sent": True, "started": True, "retryable": False,
                    "msg_id": command_msg_id,
                    "raw": text + "\n" + "\n".join(verify_raw),
                }

        logger.warning(
            "execute_cook 已下发但设备未进入运行态 cook_id=%s device=%s",
            _log_fingerprint(cook_id), _log_fingerprint(device_id or "default"),
        )
        return {
            "ok": False,
            "code": "DEVICE_STATE_UNVERIFIED",
            "reason": _device_text(lang, "push_not_running"),
            "command_sent": True,
            "started": None,
            "retryable": False,
            "msg_id": command_msg_id,
            "raw": text + "\n" + "\n".join(verify_raw),
        }

    except asyncio.TimeoutError:
        await _terminate_subprocess(proc, operation="execute_cook")
        logger.warning("execute_cook 超时 cook_id=%s", _log_fingerprint(cook_id))
        return {
            "ok": False,
            "code": "DEVICE_OUTCOME_UNKNOWN",
            "reason": _device_text(lang, "command_outcome_unknown"),
            "command_sent": None,
            "started": None,
            "retryable": False,
            "msg_id": command_msg_id,
            "raw": "",
        }
    except Exception as e:
        logger.error(
            "execute_cook 异常 cook_id=%s error_type=%s",
            _log_fingerprint(cook_id), type(e).__name__,
        )
        command_may_have_started = proc is not None
        return {
            "ok": False,
            "code": (
                "DEVICE_OUTCOME_UNKNOWN"
                if command_may_have_started
                else "EXECUTE_EXCEPTION"
            ),
            "reason": (
                _device_text(lang, "command_outcome_unknown")
                if command_may_have_started
                else _device_text(lang, "execute_exception")
            ),
            "command_sent": None if command_may_have_started else False,
            "started": None if command_may_have_started else False,
            "retryable": not command_may_have_started,
            "msg_id": command_msg_id,
            "raw": "",
        }


@trace_async_tool("device_stop")
async def stop_cook(lang: str = "zh", timeout: int = 30, device_id: str = None) -> dict:
    """确定性停止公开版 Mock：send_device_command.py stop。

    device_id 透传给脚本（不传=默认设备）。返回 {"ok": bool, "reason": str, "raw": str}。
    """
    try:
        args = ["python3", str(_SEND_COMMAND), "stop", "0", "0", "0", lang]
        if device_id:
            args.append(device_id)
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        text = out.decode(errors="replace") + err.decode(errors="replace")
        ok = '"code": 200' in text or '"success": true' in text.lower()
        if not ok:
            logger.warning(
                "stop_cook 失败 device=%s rc=%s output_chars=%s",
                _log_fingerprint(device_id or "default"), proc.returncode, len(text),
            )
        return {
            "ok": ok,
            "code": "OK" if ok else "DEVICE_STOP_REJECTED",
            "reason": "" if ok else _device_text(lang, "stop_no_ack"),
            "command_sent": True if ok else False,
            "stopped": True if ok else False,
            "raw": text,
        }
    except asyncio.TimeoutError:
        await _terminate_subprocess(proc, operation="stop_cook")
        return {
            "ok": False,
            "code": "STOP_OUTCOME_UNKNOWN",
            "reason": _device_text(lang, "command_outcome_unknown"),
            "command_sent": None,
            "stopped": None,
            "raw": "",
        }
    except Exception as e:
        logger.error("stop_cook 异常 error_type=%s", type(e).__name__)
        return {
            "ok": False,
            "code": "DEVICE_STOP_EXCEPTION",
            "reason": _device_text(lang, "execute_exception"),
            "command_sent": False,
            "stopped": False,
            "raw": "",
        }
