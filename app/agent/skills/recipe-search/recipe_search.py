"""
Recipe Search Skill - 食谱检索工具
从 Milvus 向量数据库中搜索和推荐食谱（固定返回3个结果）

用法：
  .venv/bin/python recipe_search.py "清淡的鸡肉菜"
"""
import os
import re
import hashlib
import json
import time
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv
# 加载项目根 .env：本脚本被 fast_path 当子进程调用 / 或独立运行时，也能拿到 DASHSCOPE_API_KEY 等
load_dotenv(Path(__file__).resolve().parents[4] / ".env")
from config import (
    MILVUS_CONFIG, SEARCH_CONFIG, DISPLAY_CONFIG, LOG_CONFIG, EMBEDDING_CONFIG,
    HYBRID_CONFIG, FILTER_CONFIG, RERANK_CONFIG, QU_CONFIG,
    SPLIT_LANG_ENABLED, hybrid_collection_for,
)

TRANSLATION_DICT = {
    '五花肉': 'Pork Belly', '猪肉': 'Pork', '牛肉': 'Beef', '羊肉': 'Lamb',
    '鸡肉': 'Chicken', '鸭肉': 'Duck', '鱼肉': 'Fish', '虾': 'Shrimp',
    '鸡蛋': 'Egg', '豆腐': 'Tofu', '青菜': 'Green Vegetables', '白菜': 'Cabbage',
    '西红柿': 'Tomato', '土豆': 'Potato', '胡萝卜': 'Carrot', '洋葱': 'Onion',
    '大蒜': 'Garlic', '生姜': 'Ginger', '辣椒': 'Chili', '香菇': 'Shiitake Mushroom',
    '红烧': 'Braised', '清蒸': 'Steamed', '爆炒': 'Stir-fried', '炖': 'Stewed',
    '烤': 'Roasted', '炸': 'Fried', '煮': 'Boiled', '凉拌': 'Cold Tossed',
    '糖醋': 'Sweet & Sour', '鱼香': 'Fish-flavored', '宫保': 'Kung Pao',
    '肉片': 'Sliced Meat', '肉丝': 'Shredded Meat', '肉末': 'Minced Meat',
    '排骨': 'Ribs', '鸡翅': 'Chicken Wings', '鸡腿': 'Chicken Drumsticks',
    '鸡胸肉': 'Chicken Breast', '牛肉片': 'Beef Slices', '羊肉串': 'Lamb Skewers',
    '汤': 'Soup', '粥': 'Congee', '面': 'Noodles', '饭': 'Rice',
    '家常菜': 'Home-style', '宴客': 'Party Dish', '快手': 'Quick', '简单': 'Easy',
    '低脂': 'Low-fat', '高蛋白': 'High-protein', '减肥': 'Diet', '健康': 'Healthy',
    '清淡': 'Light', '辣': 'Spicy', '微辣': 'Mildly Spicy', '中辣': 'Medium Spicy',
    '不辣': 'Non-spicy', '素食': 'Vegetarian', '早餐': 'Breakfast', '午餐': 'Lunch',
    '晚餐': 'Dinner', '中餐': 'Chinese', '川菜': 'Sichuan', '粤菜': 'Cantonese',
    '分钟': 'min'
}

try:
    import preload
except Exception:
    pass

CACHE_DIR = Path.home() / ".openclaw" / "workspace" / "state" / "query_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def translate_to_english(text: str) -> str:
    result = text
    sorted_keys = sorted(TRANSLATION_DICT.keys(), key=len, reverse=True)
    for cn in sorted_keys:
        en = TRANSLATION_DICT[cn]
        result = result.replace(cn, en)
    return result


def _config_signature() -> str:
    """检索相关配置的短签名，并入缓存 key —— 切换 hybrid/filter/rerank 配置或
    灰度时，旧配置缓存不会串味命中（endpoint/timeout 不影响结果，排除在外）。"""
    payload = json.dumps({
        "hybrid": HYBRID_CONFIG,
        "filter": FILTER_CONFIG,
        "rerank": {k: RERANK_CONFIG.get(k) for k in ("enabled", "model", "candidate_k")},
        "qu": {k: QU_CONFIG.get(k) for k in ("enabled", "model", "min_chars")},
        "split_lang": SPLIT_LANG_ENABLED,
        "result_dedup": "lang_name_v1",
        # Milvus metadata 已新增全量 recipe_detail；升级缓存版本，避免旧查询结果
        # 命中后丢失详情、时长、份量与步骤。
        "result_schema": "recipe_detail_v1_nested_record_type_filter_v2_20260731",
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


_CFG_SIG = _config_signature()


def _get_cache_key(query: str, top_k: int, lang: str = "") -> str:
    return hashlib.md5(f"{query}:{top_k}:{lang}:{_CFG_SIG}".encode()).hexdigest()


def _load_from_cache(cache_key: str, max_age_seconds: int = 3600) -> Optional[List[Dict[str, Any]]]:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if time.time() - data.get("timestamp", 0) > max_age_seconds:
                cache_file.unlink(missing_ok=True)
                return None
            # cache_key 是低熵查询的可枚举摘要，不写入日志。
            logger.info("[Cache] 命中查询缓存")
            return data.get("results")
    except Exception as e:
        logger.debug(f"缓存读取失败：error_type={type(e).__name__}")
        return None


def _save_to_cache(cache_key: str, results: List[Dict[str, Any]]):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"timestamp": time.time(), "results": results}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"缓存写入失败：error_type={type(e).__name__}")


def _build_filter_expr(is_chinese: bool, qfacets: Dict[str, Any], split_lang: bool = False) -> str:
    """按硬过滤口径拼 Milvus filter 表达式：lang(按 query 语言) + 素食 + 可选硬 facet。

    split_lang=True（分集合路由）时语言已由集合隔离，不再拼 lang 子句。"""
    clauses: List[str] = []
    if FILTER_CONFIG.get("lang_auto") and not split_lang:
        clauses.append(f'metadata["lang"] == "{"zh" if is_chinese else "en"}"')
    if FILTER_CONFIG.get("diet_hard") and qfacets.get("diet_veg"):
        clauses.append(
            '(json_contains(metadata["facets"]["diet"], "vegetarian") '
            'or json_contains(metadata["facets"]["diet"], "vegan"))'
        )
    for dim in FILTER_CONFIG.get("hard_facet_dims", []):
        for val in qfacets.get("facets", {}).get(dim, []):
            clauses.append(f'json_contains(metadata["facets"]["{dim}"], "{val}")')
    return " and ".join(clauses)


def _apply_excludes(results: List[Dict[str, Any]], excludes: List[str]) -> List[Dict[str, Any]]:
    """忌口硬过滤：剔除食材或菜名里命中忌口词的结果（子串匹配，比 Milvus 精确匹配更稳）。"""
    if not excludes:
        return results
    out = []
    for r in results:
        md = r.get("metadata", {})
        ings = md.get("ingredients", [])
        hay = ("".join(ings) if isinstance(ings, list) else str(ings)) + str(md.get("name", ""))
        if any(x in hay for x in excludes):
            continue
        out.append(r)
    return out


def _blank_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        return not s or s.lower() in {"nan", "none", "null", "未知", "未知食材"}
    if isinstance(value, (list, tuple, set)):
        return not any(not _blank_metadata_value(v) for v in value)
    if isinstance(value, dict):
        return not any(not _blank_metadata_value(v) for v in value.values())
    return False


import re as _re_placeholder

_PLACEHOLDER_INGREDIENT_RE = _re_placeholder.compile(
    r"^(?:food|ingredient|item|材料|食材|配料)\s*\d+$",
    _re_placeholder.IGNORECASE,
)

_PLACEHOLDER_NAME_RE = _re_placeholder.compile(
    r"^(?:chefyap|recipe|dish|menu)\s*\d+$",
    _re_placeholder.IGNORECASE,
)


def _has_placeholder_data(md: dict) -> bool:
    """检测占位/测试数据：菜名或食材匹配 food1/ingredient2/chefyap0430 等模式。"""
    name = str(md.get("name") or "").strip()
    if name and _PLACEHOLDER_NAME_RE.match(name):
        return True
    ingredients = md.get("ingredients")
    if isinstance(ingredients, str):
        ingredient_list = [item.strip() for item in ingredients.replace(";", ",").split(",") if item.strip()]
    elif isinstance(ingredients, (list, tuple)):
        ingredient_list = [str(item).strip() for item in ingredients if str(item).strip()]
    else:
        ingredient_list = []
    if ingredient_list:
        placeholder_count = sum(1 for item in ingredient_list if _PLACEHOLDER_INGREDIENT_RE.match(item))
        if placeholder_count >= len(ingredient_list) * 0.5 and placeholder_count >= 2:
            return True
    return False


def _apply_quality_filter(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """线上推荐质量兜底：非菜谱、缺食材、缺图片或占位数据的记录不进入最终候选。"""
    out = []
    incomplete_dropped = 0
    non_recipe_dropped = 0
    placeholder_dropped = 0
    for r in results:
        md = r.get("metadata", {})
        facets = md.get("facets") if isinstance(md.get("facets"), dict) else {}
        raw_record_types = [
            md.get("record_type"),
            facets.get("record_type"),
        ]
        record_types = {
            str(item or "").strip().lower()
            for value in raw_record_types
            for item in (
                value
                if isinstance(value, (list, tuple, set))
                else [value]
            )
            if str(item or "").strip()
        }
        if "device_program" in record_types:
            non_recipe_dropped += 1
            continue
        if _blank_metadata_value(md.get("ingredients")) or _blank_metadata_value(md.get("image_url")):
            incomplete_dropped += 1
            continue
        if _has_placeholder_data(md):
            placeholder_dropped += 1
            continue
        out.append(r)
    if non_recipe_dropped:
        logger.info(
            f"[RecipeSearch] 类型过滤剔除 {non_recipe_dropped} 条设备程序候选"
        )
    if incomplete_dropped:
        logger.info(
            f"[RecipeSearch] 质量过滤剔除 {incomplete_dropped} 条缺食材/图片候选"
        )
    if placeholder_dropped:
        logger.info(
            f"[RecipeSearch] 质量过滤剔除 {placeholder_dropped} 条占位/测试数据候选"
        )
    return out


def _normalize_result_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    return re.sub(r"\s+", "", text)


def _result_dedup_key(result: Dict[str, Any]) -> tuple:
    """推荐列表展示去重：同语言同菜名只展示一次；业务主键仍是 recipe_id + lang。"""
    md = result.get("metadata", {})
    lang = str(md.get("lang") or "").strip().lower()
    name = _normalize_result_name(md.get("name"))
    if name:
        return ("lang_name", lang, name)
    return ("business_key", str(md.get("recipe_id") or "").strip(), lang)


def _dedupe_display_results(results: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    dropped = 0
    for r in results:
        key = _result_dedup_key(r)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
        if limit is not None and len(out) >= limit:
            break
    if dropped:
        logger.info(f"[RecipeSearch] 展示去重剔除 {dropped} 条同语言同名候选")
    return out


def _normalize_scores(results: List[Dict[str, Any]], rrf_k: int) -> None:
    """把 RRF 融合分归一到 [0,1] 作展示置信度（1.0=两路都排第一）；原始分留在 rrf_score。"""
    denom = 2.0 / (rrf_k + 1)
    for r in results:
        r["rrf_score"] = r.get("score", 0.0)
        r["score"] = max(0.0, min(1.0, r["rrf_score"] / denom)) if denom else r["rrf_score"]


def _doc_for_rerank(md: Dict[str, Any]) -> str:
    """给重排模型看的文档串：菜名 + 食材 + 中文标签（英文标签去掉，省 token）。"""
    name = md.get("name", "")
    ings = md.get("ingredients", [])
    ings_str = "、".join(ings[:8]) if isinstance(ings, list) else str(ings)
    zh_tags = []
    tags = md.get("tags", [])
    if isinstance(tags, list):
        for t in tags:
            t = str(t)
            if any("一" <= c <= "鿿" for c in t) and not any("a" <= c.lower() <= "z" for c in t):
                zh_tags.append(t)
    parts = [name]
    if ings_str:
        parts.append(f"食材：{ings_str}")
    if zh_tags:
        parts.append(f"标签：{'、'.join(zh_tags[:8])}")
    return "。".join(parts)


def _rerank_results(query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """gte-rerank-v2 精排候选池 → 展示去重 top_k；失败回退召回原序。"""
    from module.reranker import get_reranker
    reranker = get_reranker(
        api_key=EMBEDDING_CONFIG.get("api_key"), model=RERANK_CONFIG["model"],
        endpoint=RERANK_CONFIG["endpoint"], timeout=RERANK_CONFIG["timeout"],
    )
    docs = [_doc_for_rerank(r["metadata"]) for r in results]
    ranked = reranker.rerank(query, docs, top_n=len(results))
    if not ranked:
        logger.info("[RecipeSearch] 重排未生效，用召回原序")
        return _dedupe_display_results(results, top_k)
    out = []
    for idx, score in ranked:
        r = results[idx]
        r["rerank_score"] = score
        r["score"] = score
        out.append(r)
    out = _dedupe_display_results(out, top_k)
    logger.info(f"[RecipeSearch] 重排完成 top{len(out)}")
    return out


@lru_cache(maxsize=100)
def search_recipes(query: str, top_k: int = 3, lang: Optional[str] = None) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = SEARCH_CONFIG["default_top_k"]

    # 语言：显式 lang（通道 locale / 调用方指定）优先，支持任意语言（分集合路由按它选集合）；
    # 缺省再回退按本次 query 自动判 zh/en。避免 keywords 被改写后误判。
    if lang:
        lang_norm = lang
    else:
        lang_norm = "zh" if _detect_language(query) else "en"
    is_zh = (lang_norm == "zh")

    logger.info(f"[RecipeSearch] 搜索开始：query_chars={len(str(query or ''))}（lang={lang_norm}）")

    cache_key = _get_cache_key(query, top_k, lang_norm)
    cached_results = _load_from_cache(cache_key)
    if cached_results:
        return cached_results

    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from module.embedding import get_embedding_model
        from dataset.milvus import get_milvus_manager
        from module.query_understanding import analyze_query

        # 1) 查询理解：口语 query → search_query + 结构化槽位（长 query 走 qwen-plus，短走规则）
        analysis = analyze_query(
            query, enabled=QU_CONFIG["enabled"], model=QU_CONFIG["model"],
            min_chars=QU_CONFIG["min_chars"], api_key=EMBEDDING_CONFIG.get("api_key"),
            base_url=EMBEDDING_CONFIG.get("base_url"), timeout=QU_CONFIG["timeout"],
        )
        search_query = analysis["search_query"]
        # 分集合路由开启且走 hybrid 时：语言已由集合隔离，filter 不再拼 lang 子句
        split = SPLIT_LANG_ENABLED and HYBRID_CONFIG.get("enabled")
        filter_expr = _build_filter_expr(is_zh, analysis, split_lang=split)
        excludes = analysis.get("exclude", []) if FILTER_CONFIG.get("exclude_hard") else []
        # 查询、facet 值和忌口都可能包含用户隐私；日志只保留诊断所需形状。
        facet_dims = sorted(str(key) for key in (analysis.get("facets") or {}).keys())
        logger.info(
            f"[RecipeSearch] QU({analysis['via']}) search_chars={len(str(search_query or ''))} "
            f"facet_dims={facet_dims} vegetarian={bool(analysis.get('diet_veg'))} "
            f"exclude_count={len(excludes)} filter_applied={bool(filter_expr)}"
        )

        # 2) 向量化（用理解后的 search_query）
        embedding_model = get_embedding_model(
            api_key=EMBEDDING_CONFIG.get("api_key"),
            base_url=EMBEDDING_CONFIG.get("base_url"),
            model=EMBEDDING_CONFIG.get("model"),
            timeout=EMBEDDING_CONFIG.get("timeout", 30),
        )
        query_vector = embedding_model.encode(search_query)[0].tolist()

        if HYBRID_CONFIG.get("enabled"):
            mgr = get_milvus_manager(
                uri=MILVUS_CONFIG.get("uri"),
                collection_name=hybrid_collection_for(lang_norm),
                vector_dim=MILVUS_CONFIG["vector_dim"],
            )
            use_rerank = RERANK_CONFIG.get("enabled")
            # 重排开启时先召回更大的候选池，再精排到 top_k
            pool_k = RERANK_CONFIG["candidate_k"] if use_rerank else top_k + (10 if excludes else 0)
            logger.info(f"[RecipeSearch] 混合召回候选池 pool_k={pool_k}（rerank={use_rerank}）...")
            results = mgr.hybrid_search(
                query_embedding=query_vector, query_text=search_query, top_k=pool_k,
                filter_expr=filter_expr, recall_k=HYBRID_CONFIG["recall_k"],
                rrf_k=HYBRID_CONFIG["rrf_k"],
            )
            results = _apply_excludes(results, excludes)
            results = _apply_quality_filter(results)
            _normalize_scores(results, HYBRID_CONFIG["rrf_k"])   # 重排失败时的兜底分数
            results = _rerank_results(search_query, results, top_k) if (use_rerank and len(results) > 1) \
                else _dedupe_display_results(results, top_k)
        else:
            # 回退纯语义（基线集合）
            mgr = get_milvus_manager(
                uri=MILVUS_CONFIG.get("uri"),
                collection_name=MILVUS_CONFIG["collection_name"],
                vector_dim=MILVUS_CONFIG["vector_dim"],
            )
            results = mgr.search(query_vector, top_k=top_k + (10 if excludes else 0),
                                 filter_expr=filter_expr)
            results = _dedupe_display_results(_apply_quality_filter(_apply_excludes(results, excludes)), top_k)

        logger.info(f"[RecipeSearch] 搜索完成，找到 {len(results)} 条结果")
        results = sorted(results, key=lambda x: (-x["score"], str(x["id"])))
        _save_to_cache(cache_key, results)
        return results

    except ImportError as e:
        logger.error(f"[RecipeSearch] 模块导入失败：error_type={type(e).__name__}")
        raise RuntimeError(f"Recipe Search 依赖模块未找到：{e}")
    except Exception as e:
        logger.error(f"[RecipeSearch] 搜索失败：error_type={type(e).__name__}")
        raise


def _detect_language(query: str) -> bool:
    chinese_chars = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    total_chars = len(query.replace(' ', ''))
    return (chinese_chars / total_chars) > 0.3 if total_chars > 0 else False


def _build_tags_str(tags: Any, is_chinese: bool) -> tuple:
    TAG_EMOJI = {
        '素食': '🌿', '素': '🌿', '蔬菜': '🌿',
        '低脂': '🔥', '减肥': '🔥', '瘦身': '🔥',
        '高蛋白': '💪', '蛋白质': '💪',
        '快手': '⚡', '快速': '⚡', '简单': '⚡',
        '炖': '🍲', '蒸': '♨️', '炒': '🍳', '烤': '🍖', '炸': '🍟', '煮': '🍜', '红烧': '🥘',
        '中餐': '🥢', '西餐': '🍴', '日料': '🍣', '东南亚菜': '🍜', '江浙菜': '🍚',
        '早餐': '🌅', '午餐': '☀️', '晚餐': '🌙',
        '甜品': '🍰', '汤': '🍲', '粥': '🥣',
        '辣': '🌶️', '清淡': '🍃', '咸鲜': '🧂', '甜': '🍯',
        '宴客': '🎉', '家常菜': '🏠', '滋补': '💊', '秋日': '🍂',
    }

    time_str = ""
    tags_str = ""
    if not tags:
        return tags_str, time_str

    tag_list = []
    if isinstance(tags, dict):
        tag_list = [v for v in tags.values() if v]
    elif isinstance(tags, list):
        tag_list = list(tags)

    tagged_items = []
    for tag in tag_list[:5]:
        time_match = re.search(r'(\d+)\s*分钟', tag)
        if time_match:
            time_str = f" ⏱️{time_match.group(1)}{'分钟' if is_chinese else ' min'}"
            continue
        emoji = ''
        for key, emo in TAG_EMOJI.items():
            if key in tag:
                emoji = emo
                break
        if not is_chinese:
            tag = translate_to_english(tag)
        tagged_items.append(f"{emoji}{tag}")

    tags_str = ' '.join(tagged_items[:4])
    return tags_str, time_str


def format_recipe_results(results: List[Dict[str, Any]], max_results: int = None, query: str = "") -> str:
    if max_results is None:
        max_results = DISPLAY_CONFIG["max_display_results"]

    if not results:
        return "未找到相关食谱，请尝试其他关键词"

    # 固定格式：按照规范要求的格式输出
    output = []
    output.append("🍳 为您找到以下食谱：\n")

    for i, result in enumerate(results[:max_results], 1):
        metadata = result.get("metadata", {})
        name = metadata.get("name", "未知菜品")
        ingredients = metadata.get("ingredients", [])
        tags = metadata.get("tags", {})
        image_url = metadata.get("image_url", "")
        
        # 处理食材列表
        if isinstance(ingredients, list):
            ingredients_str = "、".join(ingredients[:5])  # 只显示前5个食材
        elif isinstance(ingredients, str):
            ingredients_str = ingredients
        else:
            ingredients_str = "未知食材"
            
        # 处理标签
        tags_list = []
        if isinstance(tags, dict):
            tags_list = [v for v in tags.values() if v]
        elif isinstance(tags, list):
            tags_list = tags
            
        # 过滤掉时间相关的标签
        filtered_tags = []
        for tag in tags_list:
            if not re.search(r'\d+\s*分钟', tag):
                filtered_tags.append(tag)
                
        tags_str = " / ".join(filtered_tags[:3]) if filtered_tags else "未知标签"
        
        output.append(f"{i}. **{name}**")
        output.append(f"   🖼️ 图片：{image_url}")
        output.append(f"   🥘 食材：{ingredients_str}")
        output.append(f"   🏷️ 标签：{tags_str}")
        output.append("")

    # 移除最后的空行
    if output and output[-1] == "":
        output.pop()
        
    return "\n".join(output)


def recipe_search_tool(query: str, top_k: int = None, lang: str = None) -> Dict[str, Any]:
    if top_k is None:
        top_k = SEARCH_CONFIG["default_top_k"]

    try:
        results = search_recipes(query, top_k, lang=lang)
        formatted_output = format_recipe_results(results, top_k, query=query)
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
            "formatted": formatted_output
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python recipe_search.py <查询文本> [top_k] [--json]")
        print("示例：python recipe_search.py '清淡的鸡肉菜' 3")
        print("      python recipe_search.py '清淡的鸡肉菜' 3 --json")
        sys.exit(1)

    query = sys.argv[1]
    top_k = 3  # 强制固定为3个结果
    output_json = "--json" in sys.argv

    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        top_k = int(sys.argv[2])

    # 可选 --lang zh|en：由调用方（fast_path 子进程回退）按原始用户输入显式指定语言
    lang = None
    if "--lang" in sys.argv:
        _i = sys.argv.index("--lang")
        if _i + 1 < len(sys.argv):
            lang = sys.argv[_i + 1]

    result = recipe_search_tool(query, top_k, lang=lang)

    if output_json:
        # JSON 模式：输出完整结构化数据（供程序调用）
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["success"]:
        print(result["formatted"])
    else:
        print(f"搜索失败：{result['error']}")
