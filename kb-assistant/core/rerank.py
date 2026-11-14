"""Day 19: Rerank —— 混合宽召回 + Cross-Encoder 精排

链路位置: 查询 → [向量+BM25 宽召回 top-20] → RRF 融合候选池 → [Cross-Encoder 精排] → top-N 进 Context
用法: python rerank.py                       # 跑内置测试问题对比"精排前后"
      python rerank.py "你的问题"            # 查指定问题

为什么需要 Rerank（对照 rag_optimization.md §四）:
  向量/BM25 都是"各自独立看 Query 和文档"再打分，快但粗糙；
  Cross-Encoder 把 (Query, 文档) 拼在一起过同一个模型，能看到两者的深度交互，
  打分准得多 —— 所以让它只给 20 条候选精细打分（慢但量少，两全）。

判断标准（面试高频）:
  Recall@K 低（相关内容根本没召回）→ 先修召回，Rerank 没用武之地；
  召回了但排不准 → 该上 Rerank。咱现在就是后一种: BC-1 的经验值段能进池子(宽召回)，
  但融合排名不够靠前 —— 正是 Rerank 的主场。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")            # 模型均已缓存
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
from sentence_transformers import CrossEncoder
from retrieval import HybridRetriever   # 复用 Day 18: 双路检索 + RRF

RERANKER = "BAAI/bge-reranker-base"   # 中文 Cross-Encoder, 约 278M 参数, CPU 可跑
POOL = 20      # 宽召回池子大小: 精排前先从 41 片里捞 20 条候选
TOP_N = 5      # 精排后保留进 Context 的条数


def rerank(question, pool, texts, cross):
    """给候选池里每个 (Query, 文档) 对打分, 按分重排。

    注意: Cross-Encoder 输出的是原始 logit(可正可负, 不在 0~1),
    只有"相对大小"有意义 —— 比排序别比绝对值, 这是它和余弦相似度的第二个区别。
    """
    pairs = [(question, texts[idx]) for idx, _ in pool]
    scores = cross.predict(pairs)                      # 20 对, CPU 上十几秒
    ranked = sorted(
        [(idx, float(s)) for (idx, _), s in zip(pool, scores)],
        key=lambda x: -x[1],                           # 分数越高越相关
    )
    return ranked[:TOP_N]


def preview(texts, idx):
    """跳过标题路径行, 预览正文（同 hybrid_retrieval 的做法）。"""
    return texts[idx].split("\n", 1)[-1].strip()[:38].replace("\n", " ")


if __name__ == "__main__":
    model = __import__("sentence_transformers").SentenceTransformer("BAAI/bge-small-zh-v1.5")
    hr = HybridRetriever(model)                 # 双路检索器（含 BM25 索引和 chroma 连接）
    cross = CrossEncoder(RERANKER, max_length=512)   # 精排裁判

    questions = sys.argv[1:] or [
        "chunk 大小设为 64 token 是文中的什么例子",   # BM25 强项题, 看精排能否扶正
        "切片大小一般设多少合适？",                     # BC-1, 看经验值段能否守住第一
        "Embedding 和 Rerank 有什么区别",             # 基准题, 看会不会被改坏
        "怎么煮红烧肉",                               # 语料外, 看裁判给低分的表现
    ]

    for q in questions:
        # 宽召回: 两路各捞 POOL 条, RRF 融合成候选池
        pool = hr.rrf_fuse(hr.vector_search(q, POOL), hr.bm25_search(q, POOL), k=POOL)
        reranked = rerank(q, pool, hr.texts, cross)

        print(f"\n【问题】{q}")
        print("  精排前 (混合 Top-3):")
        for rank, (idx, _) in enumerate(pool[:3], 1):
            print(f"    {rank}. [{hr.sources[idx]}] {preview(hr.texts, idx)}...")
        print("  精排后 (Rerank Top-3):")
        for rank, (idx, score) in enumerate(reranked[:3], 1):
            print(f"    {rank}. [{hr.sources[idx]}] (分 {score:.2f}) {preview(hr.texts, idx)}...")
