"""v0.1+v0.2: RAG 问答 —— 混合宽召回 → Rerank 精排 → 拼 Prompt → 带引用生成

链路位置: 文档 → 切片 → 向量化 → 入库 → [混合召回 top-20 → Rerank 精排 top-K] → 生成
用法: python rag.py "问题1" ... ｜ RAG_K=3 python rag.py ...（K = 精排后进 Context 的片数）
v0.2 验收: 精排后相关小节进 Top-1；RAG_K=3 时输入 Token 明显低于纯向量 K=5 的基线。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")            # bge/reranker 已缓存，跳过联网校验
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # 关 chromadb 匿名上报

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from chat import chat      # Day 2 的地基: 超时/重试/Token 统计，chat() 内部自己 load .env
from retrieval import HybridRetriever   # Day 18: 向量+BM25 双路 + RRF 融合

MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 离线编码必须和入库用同一个模型
RERANKER = "BAAI/bge-reranker-base"     # v0.2 精排裁判（Cross-Encoder）
COLLECTION = os.environ.get("RAG_KB", "kb_ta_struct")   # 查哪个集合
POOL = 20                               # 宽召回池子大小（精排的原料）
K = int(os.environ.get("RAG_K", "5"))   # 精排后进 Context 的片数（可调小省 Token）


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


def retrieve_reranked(question, model, cross, hr, pool=POOL, k=K):
    """v0.2 检索主链路: 混合宽召回 → Rerank 精排。

    pool=20 粗筛（快而糙，两路互补兜底）→ Cross-Encoder 给 20 对 (问题, 片) 精细打分
    （慢而准，量少所以可接受）→ 取前 k 片。
    Rerank 分数是原始 logit，只有相对大小有意义——比排序别比绝对值。
    """
    fused = hr.rrf_fuse(
        hr.vector_search(question, pool),
        hr.bm25_search(question, pool),
        k=pool,
    )                                                  # [(片序号, rrf分)]
    pairs = [(question, hr.texts[idx]) for idx, _ in fused]
    scores = cross.predict(pairs)
    ranked = sorted(
        [(idx, float(s)) for (idx, _), s in zip(fused, scores)],
        key=lambda x: -x[1],
    )[:k]
    return [(hr.sources[idx], hr.texts[idx]) for idx, _ in ranked], fused


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
        "3. 如果资料不足以回答问题，明确说「资料中没有相关内容」，不要猜测或编造。"
        "判断「不足」的标准：资料必须直接回答了问题——只是出现了相关关键词、只定义了相关概念、或仅在示例中顺带提到，都不算回答了问题"
        "（例：资料定义了 RAG 和微调，但没有讲二者如何选型时，应回答「资料中没有相关内容」，而不是自编选型原则；"
        "资料里拿「今天天气」当调用示例，不代表资料掌握了天气知识）；\n"
        "4. 但推断必须有据：只有当资料给出了可用于推导的原理或数据时，才允许据此推断作答"
        "（如由「overlap 的作用是防边界割裂」推出「设为 0 则失去保护」）；"
        "推断只能沿着资料中的原理走一步，不能把概念解释扩展成资料未讲的方法论、选型建议或操作步骤。"
    )
    user = f"【资料】\n{context}\n\n【问题】{question}"

    # messages 是 week1 学过的标准格式: 对话历史 = 一个 dict 列表
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    import sys
    model = SentenceTransformer(MODEL_NAME)   # 编码器只加载一次
    cross = CrossEncoder(RERANKER, max_length=512)  # 精排裁判只加载一次
    hr = HybridRetriever(model)               # 双路检索器（含 BM25 索引）
    print(f"[配置] 宽召回 {POOL} → 精排取 {K} 片 (RAG_K 可调)")

    # 命令行传了问题就测命令行的: python rag.py "问题1" "问题2"
    # 没传就用默认验收三连
    questions = sys.argv[1:] or [
        "第六周的学习内容是什么？",
        "Day 16 修复了哪些 Bad Case？",
        "怎么煮红烧肉",
    ]

    for q in questions:
        print(f"\n{'=' * 60}\n【问题】{q}\n{'-' * 60}")

        hits, fused = retrieve_reranked(q, model, cross, hr)  # ① 宽召回+精排
        msgs = build_messages(q, hits)        # ② 拼 Prompt
        result = chat(msgs, temperature=0.1)  # ③ 生成（低温: 事实问答求稳）

        print(result["text"])
        print("-" * 60)
        print(f"Token: 输入 {result['usage']['in']} / 输出 {result['usage']['out']}")
        # 引用来源 = 本题命中的片段来源去重（即使拒答，资料也照常检索了）
        cited = sorted({s for s, _ in hits})
        print(f"精排后命中来源 ({len(hits)}/{len(fused)} 片): {', '.join(cited)}")
