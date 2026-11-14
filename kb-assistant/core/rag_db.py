"""Day 11 + Day 16: ChromaDB 建库 + 检索 Top-5

用法:
  python rag_db.py                                  # 用 output/ 的向量建 kb_chunk300 集合
  RAG_KB=kb_struct RAG_DIR=output_struct python rag_db.py   # 用新版切片建另一个集合

两套集合并存（chroma_db/ 里），随时对比新旧切片的检索质量。
关键认知: ChromaDB 只负责"存向量 + 快速找近邻"，向量仍由咱的 bge 模型自己算——
         绝不让 Chroma 用它默认的英文 embedding 模型代劳。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 模型已缓存，跳过联网校验
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # 关掉 chromadb 的匿名上报

import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION = os.environ.get("RAG_KB", "kb_ta_struct")   # 集合名：环境变量可切新旧版
DATA_DIR = os.environ.get("RAG_DIR", "index/chunks_struct")         # 向量/元数据所在目录


def get_collection():
    """拿到集合句柄。数据落盘在 chroma_db/ 文件夹，重跑程序库还在（持久化）。"""
    client = chromadb.PersistentClient(path="chroma_db")
    # cosine 空间下，返回的 distance = 1 - 余弦相似度
    # 0 = 一模一样，~1 = 无关，越大越不像 —— 和 Day 10 手写的 cosine_sim 正好互补
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def build_db():
    """入库：把 Day 10 的向量 + 原文 + 来源，一次性喂给 ChromaDB。

    三个坑藏在参数里：
    1. 不传 embedding_function → 阻止 Chroma 用默认英文模型偷偷编码
    2. documents= 原文也存一份 → 检索结果直接可读，不用再查 meta.json
    3. upsert 而不是 add → 重跑脚本不报错、不重复插入
    """
    collection = get_collection()
    vectors = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8"))

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(meta))],
        embeddings=vectors.tolist(),
        documents=[m["text"] for m in meta],
        metadatas=[{"source": m["source"]} for m in meta],
    )
    print(f"[建库] {COLLECTION} 共 {collection.count()} 条（数据来自 {DATA_DIR}/）→ 落盘 chroma_db/")


def search(question, model, k=5):
    """检索：问题 → 同一个 bge 编码 → 查 Top-K → 打出来人工看质量。"""
    collection = get_collection()
    q_vec = model.encode([question])[0].tolist()   # 必须同一个模型，才和库里向量同空间

    res = collection.query(query_embeddings=[q_vec], n_results=k)
    # 返回结构: 每个字段都是"套一层列表"(因为支持一次查多条)，咱只查了一条所以取 [0]
    ids, docs = res["ids"][0], res["documents"][0]
    metas, dists = res["metadatas"][0], res["distances"][0]

    print(f"\n[检索] {question}")
    for rank, (id_, doc, m, d) in enumerate(zip(ids, docs, metas, dists), 1):
        sim = 1 - d   # cosine 空间: 相似度 = 1 - 距离
        preview = doc[:50].replace("\n", " ")
        print(f"  {rank}. 相似度 {sim:.3f} | [{m['source']}] {id_}")
        print(f"     {preview}...")


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)   # 只加载一次，多次检索共用
    build_db()

    # 自测三连: 前两个应该命中相关切片，第三个应该"全军覆没"
    search("为什么切片不能太大也不能太小", model)
    search("Embedding 和 Rerank 有什么区别", model)
    search("怎么煮红烧肉", model)
