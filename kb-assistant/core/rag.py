"""Day 12: 最小 RAG —— 提问 → 检索 → 拼 Prompt → 带引用生成

链路位置: 文档 → 切片 → 向量化 → 入库/检索 → [今天: 拼 Prompt + 生成] ✅ 全链路闭环
用法: python rag.py
本周验收: 回答基本正确、带 [n] 引用；问语料外的问题应拒答（对抗幻觉）。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")            # bge 已缓存，跳过联网校验
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # 关 chromadb 匿名上报

import chromadb
from sentence_transformers import SentenceTransformer
from chat import chat      # Day 2 的地基: 超时/重试/Token 统计，chat() 内部自己 load .env

MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 离线编码必须和入库用同一个模型
COLLECTION = os.environ.get("RAG_KB", "kb_ta_struct")   # 查哪个集合（新旧切片版可切换）
K = 5                                    # 每次检索取 Top-5 片段


def get_collection():
    """拿集合句柄（同 rag_db.py）。数据已落盘 chroma_db/，这里直接读，不用重新入库。"""
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def retrieve(question, model, k=K):
    """在线第一步：问题 → 向量 → Top-K 片段。

    返回 [(source, text), ...]。
    注意问题必须用和建库同一个模型编码——不同模型 = 不同向量空间 = 比较失效。
    """
    collection = get_collection()
    q_vec = model.encode([question])[0].tolist()

    res = collection.query(query_embeddings=[q_vec], n_results=k)
    # 返回的每个字段都套一层列表(支持批量查询)，咱单条查询所以取 [0]
    sources = res["metadatas"][0]
    texts = res["documents"][0]
    return [(m["source"], t) for m, t in zip(sources, texts)]


def build_messages(question, hits):
    """在线第二步：把检索结果拼成符合五段式精神的 Prompt。

    结构:
      system = 角色 + 三条硬规则（只依据资料 / 标引用 / 不足直说）
      user   = 【资料】[1]-[5] 每片带编号和来源 + 【问题】

    三条规则分别对付三种翻车:
      规则1 锁死"自由发挥" —— 模型只能用检索到的内容
      规则2 让回答可溯源   —— [n] 能映射回具体文档
      规则3 给"不知道"留出口 —— 红烧肉问题必须拒答而不是硬编
    """
    # 资料块: 每片编号 + 来源文件名 + 片内容，片与片之间空一行
    blocks = []
    for i, (source, text) in enumerate(hits, 1):
        blocks.append(f"[{i}] ({source})\n{text}")
    context = "\n\n".join(blocks)

    system = (
        "你是「LLM/RAG 六周课程」的学习助教，帮学员解答课程内容、学习安排与项目问题。严格遵守以下规则：\n"
        "1. 只依据【资料】中的内容回答，不要使用资料之外的知识；\n"
        "2. 回答中的论点要标注来源编号，例如 [1]、[3]；\n"
        "3. 如果资料不足以回答问题，明确说「资料中没有相关内容」，不要猜测或编造；\n"
        "4. 但如果资料已经给出了相关原理或数据，应据此推断作答（如由「overlap 的作用是防边界割裂」推出「设为 0 则失去保护」），"
        "不要因为资料没有逐字答案就拒绝回答。"
    )
    user = f"【资料】\n{context}\n\n【问题】{question}"

    # messages 是 week1 学过的标准格式: 对话历史 = 一个 dict 列表
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    import sys
    model = SentenceTransformer(MODEL_NAME)   # 模型只加载一次

    # 命令行传了问题就测命令行的: python rag.py "问题1" "问题2"
    # 没传就用默认验收三连
    questions = sys.argv[1:] or [
        "第六周的学习内容是什么？",
        "Day 16 修复了哪些 Bad Case？",
        "怎么煮红烧肉",
    ]

    for q in questions:
        print(f"\n{'=' * 60}\n【问题】{q}\n{'-' * 60}")

        hits = retrieve(q, model)             # ① 检索
        msgs = build_messages(q, hits)        # ② 拼 Prompt
        result = chat(msgs, temperature=0.1)  # ③ 生成（低温: 事实问答求稳）

        print(result["text"])
        print("-" * 60)
        print(f"Token: 输入 {result['usage']['in']} / 输出 {result['usage']['out']}")
        # 引用来源 = 本题命中的片段来源去重（即使拒答，资料也照常检索了）
        cited = sorted({s for s, _ in hits})
        print(f"本次检索命中的来源: {', '.join(cited)}")
