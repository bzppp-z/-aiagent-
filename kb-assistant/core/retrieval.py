"""Day 18: 混合检索 —— BM25 关键词 + 向量，RRF 融合

链路位置: 查询 → [向量检索 + BM25 检索 双路并行] → RRF 融合 → (Day 19 Rerank 精排) → 生成
用法: python hybrid_retrieval.py            # 跑内置测试问题对比三路结果
      RAG_KB=kb_struct python hybrid_retrieval.py "你的问题"   # 查指定问题

核心认知:
  向量检索懂语义但模糊（"设多少"≈"经验值"，也能把不相关的算成 0.5）；
  BM25 认字不懂义，但对数字/术语/专有名词等精确词极其敏感。
  两路各查一遍，用 RRF 把排名融在一起 —— 互为兜底。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")            # bge 已缓存
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # 关 chromadb 上报

import sys
import json
import jieba
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION = os.environ.get("RAG_KB", "kb_ta_struct")
DATA_DIR = os.environ.get("RAG_DIR", "index/chunks_struct")
RRF_K = 60        # RRF 平滑常数：越大，排名差距对融合分的影响越平缓（论文经验值）


def tokenize(text):
    """BM25 的前置工序：中文必须先分词。jieba 搜索引擎模式，兼顾粒度。"""
    return list(jieba.cut_for_search(text))


class HybridRetriever:
    """双路检索器：向量走 Chroma，BM25 走内存倒排，融合走 RRF。

    关键设计：两路用同一个"片序号"体系（meta.json 的下标），
    融合时才能对得上号。
    """

    def __init__(self, model):
        # ---- 路 ② 的地基: BM25 倒排索引 ----
        meta = json.load(open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8"))
        self.meta = meta
        self.texts = [m["text"] for m in meta]
        self.sources = [m["source"] for m in meta]
        tokenized = [tokenize(t) for t in self.texts]      # 每片切成词列表
        self.bm25 = BM25Okapi(tokenized)                   # 建 BM25 索引（纯内存，秒级）

        # ---- 路 ① 的地基: 向量集合 ----
        client = chromadb.PersistentClient(path="chroma_db")
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.model = model

    # ---------- 路 ①: 向量检索 ----------
    def vector_search(self, question, k=5):
        """语义检索。返回 [(片序号, 相似度), ...] 按相似度降序。"""
        q_vec = self.model.encode([question])[0].tolist()
        res = self.collection.query(query_embeddings=[q_vec], n_results=k)
        hits = []
        for id_, d in zip(res["ids"][0], res["distances"][0]):
            idx = int(id_.split("_")[1])       # id 形如 chunk_17 → 序号 17
            hits.append((idx, 1 - d))          # cosine 空间: 相似度 = 1 - 距离
        return hits

    # ---------- 路 ②: BM25 关键词检索 ----------
    def bm25_search(self, question, k=5):
        """关键词检索。BM25 给全库每片打分，取 Top-K（只留得分>0 的）。"""
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(idx, float(s)) for idx, s in ranked[:k] if s > 0]

    # ---------- 路 ③: RRF 融合 ----------
    def rrf_fuse(self, *ranked_lists, k=5):
        """Reciprocal Rank Fusion: 对每个候选，累加各路给它的 1/(RRF_K + 名次)。

        性质: 不需要两路的分数可比（BM25 分数和余弦相似度根本不同量纲），
        只用"名次"融合，天然归一化 —— 这就是 RRF 比"分数加权"省心的原因。
        """
        fused = {}
        for hits in ranked_lists:
            for rank, (idx, _score) in enumerate(hits, 1):
                fused.setdefault(idx, 0.0)
                fused[idx] += 1.0 / (RRF_K + rank)    # 名次越靠前贡献越大
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        return ranked[:k]

    # ---------- 三路对比输出 ----------
    def show(self, question, k=5):
        vec = self.vector_search(question, k)
        kw = self.bm25_search(question, k)
        hyb = self.rrf_fuse(vec, kw, k=k)   # k 必须用关键字传: *ranked_lists 会吞掉位置参数

        print(f"\n【问题】{question}")
        for name, hits in (("向量", vec), ("BM25", kw), ("RRF融合", hyb)):
            print(f"  {name}:")
            for rank, (idx, score) in enumerate(hits, 1):
                # 跳过第一行标题路径，预览正文——否则每片看起来都一样
                body = self.texts[idx].split("\n", 1)[-1].strip()
                preview = body[:38].replace("\n", " ")
                print(f"    {rank}. [{self.sources[idx]}] (分 {score:.4f}) {preview}...")


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    hr = HybridRetriever(model)

    questions = sys.argv[1:] or [
        # 关键词强的: 精确数字/术语, 考 BM25 兜底
        "chunk 大小设为 64 token 是文中的什么例子",
        "Bi-Encoder 和 Cross-Encoder 有什么区别",
        # 语义强的: 问法与用词不同, 考向量兜底
        "切片大小一般设多少合适？",
        "怎么煮红烧肉",
    ]
    for q in questions:
        hr.show(q)
