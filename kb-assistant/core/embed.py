"""Day 10 + Day 16: 批量向量化 + 余弦相似度实验

用法:
  python embed.py                  # 向量化 output/ 里 chunk_size=300 的切片 → output/vectors.npy
  python embed.py output_struct    # 向量化指定目录里的全部切片 → output_struct/vectors.npy

产出: 目录下的 vectors.npy（向量矩阵）+ meta.json（每条向量对应的原文，顺序一一对应）
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 模型已缓存到本地；设离线跳过联网校验，否则国内连不上 HF 会干等几分钟
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 实测输出 512 维（384 是英文版 small 的，别记混）


def cosine_sim(a, b):
    """余弦相似度：方向越一致越接近 1，无关接近 0。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_texts(texts, model):
    """批量向量化：一次喂一个列表，比逐条调用快。返回 (条数, 512) 矩阵。"""
    return model.encode(texts)


def load_chunks(output_dir, chunk_size=None):
    """读目录里的切片 json，摊平成一个列表。

    chunk_size 给定就只留该版本（如 output/ 里 chunk100 和 chunk300 混着），
    不给就全要（output_struct/ 里只有一版）。
    每条记录: {"source": 来源文件名, "text": 片内容}
    """
    chunks = []
    for name in os.listdir(output_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(output_dir, name), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "chunks" not in data:
            continue  # 跳过目录里的历史产物（meta.json 等）——别吃自己的输出
        if chunk_size is not None and data["chunk_size"] != chunk_size:
            continue
        for piece in data["chunks"]:
            chunks.append({"source": data["source"], "text": piece["text"]})
    return chunks


if __name__ == "__main__":
    # 目录可选参数: 默认 output/（Day 9 的固定长度版）
    folder = sys.argv[1] if len(sys.argv) > 1 else "index/chunks_fix"
    # chunk_size 过滤可选: output/ 里混着 chunk100/chunk300，默认只要 300
    size_filter = int(sys.argv[2]) if len(sys.argv) > 2 else (300 if folder == "index/chunks_fix" else None)

    model = SentenceTransformer(MODEL_NAME)

    # ---- 实验一: 三句话感受语义相似度（词不同，意思近）----
    sents = ["如何申请退货退款", "退货流程是什么", "今天天气真好"]
    vecs = embed_texts(sents, model)
    print(f"[实验一] 模型输出维度: {vecs.shape[1]}")
    print(f"  退货退款 vs 退货流程: {cosine_sim(vecs[0], vecs[1]):.3f}")
    print(f"  退货退款 vs 今天天气: {cosine_sim(vecs[0], vecs[2]):.3f}")

    # ---- 实验二: 把切片全部向量化 ----
    chunks = load_chunks(folder, size_filter)
    print(f"[实验二] {folder}/ 共 {len(chunks)} 片，开始向量化...")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, model)
    print(f"  向量矩阵形状: {vectors.shape}")   # (片数, 512)

    # 向量和原文存进同一个目录 —— 第 i 个向量 ↔ 第 i 条 meta，顺序绝不能乱
    np.save(os.path.join(folder, "vectors.npy"), vectors)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"  已保存: {folder}/vectors.npy + {folder}/meta.json")
    print(f"  抽查 meta 第 1 条: [{chunks[0]['source']}] {chunks[0]['text'][:40]}...")
