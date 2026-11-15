"""Day 24 上半场: 检索指标评测 —— Recall@5 / MRR / 分组命中 / 拒答阈值证据

链路位置: golden_set.json → [逐题跑生产检索链路] → 指标报告
用法: python eval/run_eval.py            # 跑全量 fact+reasoning（检索指标）+ outside（阈值证据）

判定规则（尺子的刻度，改动前先想清楚）:
  Recall@5   —— fact/reasoning 题: Top-5 的来源文档与该题 source 列表有交集即命中（文档级）
  MRR        —— 第一处命中排名的倒数求平均（第 1 名=1.0，第 3 名=0.33，没中=0）
  阈值证据   —— outside 题: 记录候选池最高精排分；若普遍 ≈0 而 in-corpus ≈1，
                "Rerank 分数拒答阈值"（悬案③）即成立
注意: 本脚本只看 source（出处），不看 answer——所以未校对的条目不影响检索指标，
     但会影响 Day 24 下半场的 LLM 裁判。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
import json
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))   # 让 eval 能 import core 里的模块

from sentence_transformers import SentenceTransformer, CrossEncoder
from retrieval import HybridRetriever
from embed import MODEL_NAME
from rag import RERANKER, POOL

K = 5                                   # 与生产链路一致（rag.py 默认 RAG_K=5）
GOLD = os.path.join(ROOT, "eval", "golden_set.json")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")


def topk_with_scores(question, hr, cross, k=K, pool=POOL):
    """跑生产同款链路（混合宽召回→精排），但保留分数——rag.py 的版本不返回分数。"""
    fused = hr.rrf_fuse(hr.vector_search(question, pool), hr.bm25_search(question, pool), k=pool)
    pairs = [(question, hr.texts[idx]) for idx, _ in fused]
    scores = cross.predict(pairs)
    ranked = sorted(
        [(idx, float(s)) for (idx, _), s in zip(fused, scores)],
        key=lambda x: -x[1],
    )[:k]
    return [(hr.sources[idx], hr.texts[idx], s) for idx, s in ranked], fused


def main():
    gold = json.load(open(GOLD, encoding="utf-8"))
    cases = gold["cases"]

    model = SentenceTransformer(MODEL_NAME)
    cross = CrossEncoder(RERANKER, max_length=512)
    hr = HybridRetriever(model)

    detail, outside_rows = [], []
    for c in cases:
        top, fused = topk_with_scores(c["question"], hr, cross)
        top_sources = [s for s, _, _ in top]

        if c["type"] == "outside":
            # 悬案③: 语料外题不考命中，考"池内最高精排分"——将来做拒答阈值
            outside_rows.append({
                "id": c["id"], "question": c["question"],
                "max_rerank_score": max(s for _, _, s in top),
                "top1_source": top_sources[0],
            })
            continue

        want = set(c["source"])
        hit_rank = next((i for i, s in enumerate(top_sources, 1) if s in want), None)
        detail.append({
            "id": c["id"], "type": c["type"],
            "recall_hit": hit_rank is not None,
            "first_hit_rank": hit_rank,
            "rrf_pool_size": len(fused),
            "top5_sources": top_sources,
            "expected": c["source"],
        })
        mark = f"HIT@{hit_rank}" if hit_rank else "MISS"
        print(f"  {c['id']} [{c['type']:9s}] {mark:6s} 期望:{','.join(want)}")

    # ---- 汇总 ----
    hits = [d for d in detail if d["recall_hit"]]
    overall = len(hits) / len(detail)
    mrr = sum(1 / d["first_hit_rank"] for d in detail if d["first_hit_rank"]) / len(detail)
    by_type = {}
    for tp in ("fact", "reasoning"):
        sub = [d for d in detail if d["type"] == tp]
        by_type[tp] = sum(d["recall_hit"] for d in sub) / len(sub)
    miss = [d for d in detail if not d["recall_hit"]]

    print(f"\n===== 检索指标（生产链路: 混合宽召回{POOL}+精排，K={K}） =====")
    print(f"Recall@{K} 总体: {overall:.1%} ({len(hits)}/{len(detail)})")
    print(f"MRR: {mrr:.3f}")
    for tp, v in by_type.items():
        print(f"  {tp}: {v:.1%}")
    print(f"未命中明细: {[(d['id'], d['expected']) for d in miss] or '无'}")

    if outside_rows:
        scores = [r["max_rerank_score"] for r in outside_rows]
        print(f"\n===== 拒答阈值证据（outside {len(outside_rows)} 题） =====")
        print(f"池内最高精排分: max={max(scores):.3f} / min={min(scores):.3f} / 均值={sum(scores)/len(scores):.3f}")
        print("（对照: 语料内强相关 ≈1.0。若 outside 普遍 ≈0，阈值方案成立 → 悬案③）")
        for r in outside_rows:
            print(f"  {r['id']}  max={r['max_rerank_score']:.3f}  Top-1: [{r['top1_source']}]")

    # ---- 报告落盘（Day 26 错误分析 / 优化前后 diff 用）----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "date": str(datetime.date.today()),
        "config": {"pipeline": "hybrid+rerank", "pool": POOL, "k": K, "cases": len(cases)},
        "retrieval": {"recall_at_5": overall, "mrr": mrr, "by_type": by_type, "detail": detail},
        "outside_threshold_evidence": outside_rows,
    }
    out = os.path.join(RESULTS_DIR, f"retrieval_report_{report['date']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[报告] 已保存 {out}（eval/results/ 已入 .gitignore，属可再生成产物）")


if __name__ == "__main__":
    main()
