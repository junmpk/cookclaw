"""
Query Understanding — qwen-plus 把口语化请求解析成结构化检索意图。

产出（与规则版 normalize.extract_query_facets 同构，多一个 search_query）：
  {search_query, facets{dim:[canonical]}, diet_veg, exclude, via}

设计原则：**LLM 绝不阻断检索**。未启用 / 短 query / 超时 / 异常 / 字段非法，
都回退到规则版（extract_query_facets）。facet 值统一过受控词表归一，保证能命中
metadata.facets 过滤。
"""
import json
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from loguru import logger

try:  # 兼容被当作 module.query_understanding 导入 / 同目录直跑
    from module.normalize import FACET_VOCAB, extract_query_facets
except ImportError:  # pragma: no cover
    from normalize import FACET_VOCAB, extract_query_facets

_DIMS = ["cuisine", "main_ingredient", "flavor", "method", "scene", "meal"]
_DIM_LABEL = {"cuisine": "菜系", "main_ingredient": "主料", "flavor": "口味",
              "method": "烹饪", "scene": "场景", "meal": "餐次"}


def _vocab_hint() -> str:
    lines = []
    for dim in _DIMS:
        vals = "/".join(FACET_VOCAB[dim].keys())
        lines.append(f"{_DIM_LABEL[dim]}({dim}): {vals}")
    return "\n".join(lines)


_SYS_PROMPT = """你是菜谱搜索的查询理解模块。把用户口语化请求解析成结构化检索意图，**只输出 JSON**：
{"search_query":"用于检索的核心查询(菜名/食材/做法/口味关键词,去掉寒暄,必要时补常见同义词,<=20字)",
 "cuisine":[],"main_ingredient":[],"flavor":[],"method":[],"scene":[],"meal":[],
 "vegetarian":false,"exclude":[]}
各维度尽量从下列取值中选，没有就留空数组：
""" + _vocab_hint() + """
vegetarian：用户明确要素食/纯素才 true。
exclude：用户明确不要/忌口的食材，**拆成单个词**（如"不吃香菜和花生"→["香菜","花生"]）。
search_query 不要带寒暄、人称、时间等无关词。
必须输出 canonical code；stir_fried、deep_fried、fried 是三个不同做法，禁止用子串混淆。"""


def _canonicalize(dim: str, values: Any) -> List[str]:
    """把 LLM 给的 facet 值映射到受控词表 canonical；命中不了的丢弃。"""
    keys = FACET_VOCAB.get(dim, {})
    out: List[str] = []
    for v in values or []:
        v = str(v).strip()
        if not v:
            continue
        if v in keys:
            out.append(v)
            continue
        normalized = re.sub(r"[\s_-]+", " ", v.casefold())
        for canon, syns in keys.items():
            aliases = [canon, *syns]
            if any(re.sub(r"[\s_-]+", " ", str(alias).casefold()) == normalized for alias in aliases):
                out.append(canon)
                break
    return list(dict.fromkeys(out))


class QueryUnderstanding:
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-plus", timeout: int = 8):
        # trust_env=False：直连 DashScope，忽略本机代理（与 embedding 一致）
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout,
                             http_client=httpx.Client(trust_env=False))
        self.model = model

    def understand(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": _SYS_PROMPT},
                          {"role": "user", "content": query}],
                temperature=0.1, max_tokens=300,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"[QU] LLM 解析失败，回退规则：error_type={type(e).__name__}")
            return None

        facets: Dict[str, List[str]] = {}
        for dim in _DIMS:
            c = _canonicalize(dim, data.get(dim))
            if c:
                facets[dim] = c
        search_query = str(data.get("search_query") or "").strip() or query
        exclude = [str(x).strip() for x in (data.get("exclude") or []) if str(x).strip()]
        return {
            "search_query": search_query,
            "facets": facets,
            "diet_veg": bool(data.get("vegetarian")),
            "exclude": exclude,
            "via": "llm",
        }


_qu: Optional[QueryUnderstanding] = None


def _get_qu(api_key: str, base_url: str, model: str, timeout: int) -> QueryUnderstanding:
    global _qu
    if _qu is None:
        _qu = QueryUnderstanding(api_key=api_key, base_url=base_url, model=model, timeout=timeout)
    return _qu


def analyze_query(query: str, *, enabled: bool, model: str, min_chars: int,
                  api_key: str, base_url: str, timeout: int = 8) -> Dict[str, Any]:
    """统一查询理解入口：长/口语 query 走 LLM，短 query 或 LLM 失败走规则。"""
    q = (query or "").strip()
    if enabled and api_key and len(q) >= min_chars:
        llm = _get_qu(api_key, base_url, model, timeout).understand(q)
        if llm:
            return llm
    rf = extract_query_facets(q)
    return {"search_query": q, "facets": rf["facets"], "diet_veg": rf["diet_veg"],
            "exclude": rf["exclude"], "via": "rule"}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import EMBEDDING_CONFIG, QU_CONFIG

    for q in ["晚上想吃点清淡的，老人小孩都能吃的家常菜", "来个下饭辣菜，不吃香菜和花生", "红烧肉"]:
        a = analyze_query(q, enabled=QU_CONFIG["enabled"], model=QU_CONFIG["model"],
                          min_chars=QU_CONFIG["min_chars"], api_key=EMBEDDING_CONFIG["api_key"],
                          base_url=EMBEDDING_CONFIG["base_url"], timeout=QU_CONFIG["timeout"])
        print(f"[{a['via']}] {q!r} -> {json.dumps(a, ensure_ascii=False)}")
