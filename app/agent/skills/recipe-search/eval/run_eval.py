"""
检索评测：纯语义(baseline) vs 混合召回(hybrid)，用数字说话。

指标（known-item retrieval，已知菜品检索）：
  - recall@3 ：目标菜是否落在前 3
  - MRR@3   ：目标菜命中排名的倒数均值（越大越靠前）

种子集：首次运行自动从库里采样 N 道菜，生成 query=菜名、expect=该菜，
        写到自定义 JSON 文件（可手动替换成真实 query 日志后再跑）。

用法（技能目录下）：
  .venv/bin/python eval/run_eval.py            # 跑评测（默认 50 条）
  .venv/bin/python eval/run_eval.py --n 100    # 重新采样 100 条种子
  .venv/bin/python eval/run_eval.py --regen     # 强制重新采样
"""
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

_SKILL = Path(__file__).resolve().parent.parent
load_dotenv(_SKILL.parents[3] / ".env")
sys.path.insert(0, str(_SKILL))

from config import MILVUS_CONFIG, EMBEDDING_CONFIG, RERANK_CONFIG  # noqa: E402
from module.embedding import get_embedding_model  # noqa: E402
from module.reranker import get_reranker  # noqa: E402
from dataset.milvus import MilvusManager  # noqa: E402
from recipe_search import _doc_for_rerank  # noqa: E402  复用生产侧的文档表示

SEED_PATH = Path(__file__).resolve().parent / "seed_queries.example.json"

# 调味料停用词：构造"食材 query"时剔除，保留有区分度的主料
SEASONINGS = {
    "盐", "食盐", "油", "植物油", "食用油", "猪油", "黄油", "糖", "白砂糖", "冰糖", "蜂蜜",
    "生抽", "老抽", "酱油", "蔬菜精", "味精", "鸡精", "水", "清水", "纯净水", "高汤",
    "淀粉", "料酒", "蚝油", "醋", "陈醋", "白醋", "香醋", "葱", "小葱", "大葱", "香葱",
    "姜", "生姜", "蒜", "大蒜", "胡椒粉", "白胡椒", "黑胡椒", "花椒", "八角", "香油",
    "芝麻油", "蛋清", "蛋液", "辣椒", "干辣椒", "小米椒", "豆瓣酱", "番茄酱",
}


def _names_and_ings(hyb: MilvusManager):
    rows = hyb.client.query(hyb.collection_name, filter='metadata["lang"] == "zh"',
                            output_fields=["metadata"], limit=100000)
    out = []
    for r in rows:
        md = r["metadata"]
        md = json.loads(md) if isinstance(md, str) else md
        md = md or {}
        nm = (md.get("name") or "").strip()
        ings = md.get("ingredients") or []
        if nm:
            out.append((nm, ings if isinstance(ings, list) else []))
    return out


def build_seed(hyb: MilvusManager, n: int):
    """采样 n 道中文菜，每道造两类 query：
       - name : 菜名（已知项检索，简单基线）
       - ing  : 由主料拼成的"我有这些食材"口语 query（贴近真实、能区分检索器）
    """
    seen, items = set(), []
    for nm, ings in _names_and_ings(hyb):
        if nm not in seen:
            seen.add(nm)
            items.append((nm, ings))
    items.sort(key=lambda x: x[0])
    if not items:
        return []
    step = max(1, len(items) // n)
    sampled = items[::step][:n]
    seed = []
    for nm, ings in sampled:
        seed.append({"qtype": "name", "query": nm, "expect": nm})
        distinct = [i for i in ings if i and i not in SEASONINGS]
        if len(distinct) >= 2:
            seed.append({"qtype": "ing", "query": " ".join(distinct[:3]), "expect": nm})
    SEED_PATH.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
    return seed


def load_seed(hyb, n, regen):
    if SEED_PATH.exists() and not regen:
        return json.loads(SEED_PATH.read_text(encoding="utf-8"))
    return build_seed(hyb, n)


def evaluate(seed, emb, base, hyb, reranker=None, k=3):
    # agg[qtype][method] = {hit, mrr, n}
    methods = ["baseline", "hybrid"] + (["rerank"] if reranker else [])
    cand_k = max(k, RERANK_CONFIG["candidate_k"])
    agg = {}
    for item in seed:
        q, expect, qt = item["query"], item["expect"], item.get("qtype", "all")
        qv = emb.encode(q)[0].tolist()
        cands = hyb.hybrid_search(qv, q, top_k=cand_k)        # 同一候选池
        runs = {
            "baseline": [r["metadata"]["name"] for r in base.search(qv, top_k=k)],
            "hybrid": [r["metadata"]["name"] for r in cands[:k]],
        }
        if reranker:
            ranked = reranker.rerank(q, [_doc_for_rerank(r["metadata"]) for r in cands], top_n=k)
            runs["rerank"] = ([cands[i]["metadata"]["name"] for i, _ in ranked]
                              if ranked else runs["hybrid"])
        for bucket in (qt, "总体"):
            agg.setdefault(bucket, {m: {"hit": 0, "mrr": 0.0, "n": 0} for m in methods})
            for m in methods:
                names = runs[m]
                agg[bucket][m]["n"] += 1
                if expect in names:
                    agg[bucket][m]["hit"] += 1
                    agg[bucket][m]["mrr"] += 1.0 / (names.index(expect) + 1)
    return agg


_QLABEL = {"name": "菜名(已知项)", "ing": "食材口语(贴近真实)", "总体": "总体"}
_MLABEL = {"baseline": "纯语义", "hybrid": "混合召回", "rerank": "混合+重排"}


def main():
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 40
    regen = "--regen" in args
    no_rerank = "--no-rerank" in args

    emb = get_embedding_model(api_key=EMBEDDING_CONFIG["api_key"],
                              base_url=EMBEDDING_CONFIG["base_url"],
                              model=EMBEDDING_CONFIG["model"])
    uri = MILVUS_CONFIG["uri"]
    base = MilvusManager(uri=uri, collection_name=MILVUS_CONFIG["collection_name"])
    hyb = MilvusManager(uri=uri, collection_name=MILVUS_CONFIG["hybrid_collection"])
    reranker = None
    if RERANK_CONFIG.get("enabled") and not no_rerank:
        reranker = get_reranker(api_key=EMBEDDING_CONFIG["api_key"], model=RERANK_CONFIG["model"],
                                endpoint=RERANK_CONFIG["endpoint"], timeout=RERANK_CONFIG["timeout"])

    seed = load_seed(hyb, n, regen)
    print(f"种子集：{len(seed)} 条（{SEED_PATH.name}）  指标：recall@3 / MRR@3"
          f"{'  [含重排]' if reranker else ''}\n")

    agg = evaluate(seed, emb, base, hyb, reranker=reranker)
    for bucket in ("name", "ing", "总体"):
        if bucket not in agg:
            continue
        b = agg[bucket]
        cnt = b["baseline"]["n"]
        print(f"◆ {_QLABEL.get(bucket, bucket)}  (n={cnt})")
        print(f"  {'方案':<8}{'recall@3':>11}{'MRR@3':>10}")
        for m in ("baseline", "hybrid", "rerank"):
            if m not in b:
                continue
            rc = b[m]["hit"] / cnt if cnt else 0
            mrr = b[m]["mrr"] / cnt if cnt else 0
            print(f"  {_MLABEL[m]:<8}{rc:>11.1%}{mrr:>10.3f}")
        if cnt:
            dh = (b["hybrid"]["hit"] - b["baseline"]["hit"]) / cnt
            line = f"  Δ 混合-基线 {dh:>+7.1%}"
            if "rerank" in b:
                dr = (b["rerank"]["hit"] - b["hybrid"]["hit"]) / cnt
                line += f"    Δ 重排-混合 {dr:>+7.1%}"
            print(line + "\n")


if __name__ == "__main__":
    main()
