"""
recipe_id → 设备食谱编号（device_code）映射。

运行时从 JSON 文件加载，单例缓存。
供检索结果注入 device_code 和设备执行层使用。
"""
import json
from pathlib import Path

from loguru import logger

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MAPPING_FILE = _DATA_DIR / "id_to_code.json"

_ID_TO_CODE: dict[str, int] | None = None


def load_mapping(path: Path | None = None) -> dict[str, int]:
    """加载 id→code 映射，单例缓存。"""
    global _ID_TO_CODE
    if _ID_TO_CODE is not None:
        return _ID_TO_CODE

    file_path = path or _MAPPING_FILE
    if not file_path.exists():
        logger.warning(f"id→code 映射文件不存在: {file_path}，device_code 将为空")
        _ID_TO_CODE = {}
        return _ID_TO_CODE

    with open(file_path, "r", encoding="utf-8") as f:
        _ID_TO_CODE = json.load(f)

    logger.info(f"已加载 {len(_ID_TO_CODE)} 条 id→code 映射")
    return _ID_TO_CODE


def get_device_code(milvus_id: int | str) -> int | None:
    """给定 Milvus auto-ID，返回设备 code；无映射返回 None。"""
    mapping = load_mapping()
    return mapping.get(str(milvus_id))


def reload_mapping(path: Path | None = None) -> dict[str, int]:
    """强制重新加载映射（测试/热更新用）。"""
    global _ID_TO_CODE
    _ID_TO_CODE = None
    return load_mapping(path)
