"""
查询理解(P2)专项评测：口语化 query 上，QU 关 vs QU 开 的 recall@3。

方法：把食材种子(ing 档)包成口水话(messy)，对比三种：
  - 干净   ：原始"食材"query（recall 上限参照）
  - 口语·QU关：直接拿 messy query 检索（无 LLM 理解）
  - 口语·QU开：messy → qwen-plus 理解出 search_query 再检索
三者都走同一套 hybrid+rerank（不加过滤，隔离 query 字符串本身的影响）。

用法：.venv/bin/python eval/run_eval_qu.py [--n 30]
"""
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

_SKILL = Path(__file__).resolve().parent.parent
load_dotenv(_SKILL.parents[3] / ".env")
sys.path.insert(0, str(_SKILL))

from config import MILVUS_CONFIG, EMBEDDING_CONFIG, RERANK_CONFIG, QU_CONFIG  # noqa: E402
from module.embedding import get_embedding_model  # noqa: E402
from module.reranker import get_reranker  # noqa: E402
from module.query_understanding import analyze_query  # noqa: E402
from dataset.milvus import MilvusManager  # noqa: E402
from recipe_search import _doc_for_rerank  # noqa: E402

SEED_PATH = Path(__file__).resolve().parent / "seed_queries.example.json"

TEMPLATES = [
    "晚上想随便弄个用{}的家常菜，不要太麻烦",
    "帮我看看{}能做出什么好吃的呀",
    "中午就想吃点{}做的，简单点的",
    "家里还剩{}，求个靠谱做法",
    "下班好累，{}有没有什么快手菜",
]


def main():
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 30

    emb = get_embedding_model(api_key=EMBEDDING_CONFIG["api_key"], base_url=EMBEDDING_CONFIG["base_url"],
                              model=EMBEDDING_CONFIG["model"])
    hyb = MilvusManager(uri=MILVUS_CONFIG["uri"], collection_name=MILVUS_CONFIG["hybrid_collection"])
    rr = get_reranker(api_key=EMBEDDING_CONFIG["api_key"], model=RERANK_CONFIG["model"],
                      endpoint=RERANK_CONFIG["endpoint"], timeout=RERANK_CONFIG["timeout"])

    def retrieve(sq, k=3):
        qv = emb.encode(sq)[0].tolist()
        cands = hyb.hybrid_search(query_embedding=qv, query_text=sq, top_k=RERANK_CONFIG["candidate_k"])
        ranked = rr.rerank(sq, [_doc_for_rerank(c["metadata"]) for c in cands], top_n=k)
        if ranked:
            return [cands[i]["metadata"]["name"] for i, _ in ranked]
        return [c["metadata"]["name"] for c in cands[:k]]

    seed = [s for s in json.loads(SEED_PATH.read_text(encoding="utf-8")) if s.get("qtype") == "ing"][:n]
    agg = {"clean": 0, "messy_off": 0, "messy_on": 0}
    total = len(seed)
    for i, s in enumerate(seed):
        ings, expect = s["query"], s["expect"]
        messy = TEMPLATES[i % len(TEMPLATES)].format(ings.replace(" ", "、"))
        # QU 开：messy → search_query
        on_sq = analyze_query(messy, enabled=True, model=QU_CONFIG["model"], min_chars=0,
                              api_key=EMBEDDING_CONFIG["api_key"], base_url=EMBEDDING_CONFIG["base_url"],
                              timeout=QU_CONFIG["timeout"])["search_query"]
        if expect in retrieve(ings):
            agg["clean"] += 1
        if expect in retrieve(messy):
            agg["messy_off"] += 1
        if expect in retrieve(on_sq):
            agg["messy_on"] += 1

    print(f"口语化 QU 专项评测：{total} 条（ing 种子包成口水话）  指标：recall@3\n")
    print(f"  {'场景':<22}{'recall@3':>10}")
    print("  " + "-" * 32)
    print(f"  {'干净(原食材query)':<20}{agg['clean']/total:>10.1%}")
    print(f"  {'口语·QU关':<22}{agg['messy_off']/total:>10.1%}")
    print(f"  {'口语·QU开':<22}{agg['messy_on']/total:>10.1%}")
    print("  " + "-" * 32)
    print(f"  {'Δ QU(开-关)':<22}{(agg['messy_on']-agg['messy_off'])/total:>+10.1%}")


if __name__ == "__main__":
    main()
