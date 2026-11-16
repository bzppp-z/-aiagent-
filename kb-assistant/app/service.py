"""服务层：把 core 的检索+生成包成一次「问答事务」

职责边界：本文件只管「输入问题 → 输出 (answer, sources, usage)」，
HTTP 相关（状态码、序列化）留给 main.py——服务层不 import fastapi，便于单测。
"""
import os
import sys

# ⚠️ 顺序即初始化：这两个环境变量必须在 import transformers 之前设置，
# 否则启动时会去连 HuggingFace 官网校验模型（国内连不通，重试几分钟后崩）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# 让 app 包能 import 同级 core/ 里的模块（rag/retrieval/chat/embed）
CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")
sys.path.insert(0, CORE)

from sentence_transformers import SentenceTransformer, CrossEncoder
from retrieval import HybridRetriever
from rag import retrieve_reranked, build_messages, RERANKER
from chat import chat, client   # client 是 chat.py 里的全局 OpenAI 客户端，流式要用
from embed import MODEL_NAME


class RAGService:
    """重资源（模型/索引）持有者——由 lifespan 在启动时实例化一次，全请求共享。"""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.cross = CrossEncoder(RERANKER, max_length=512)
        self.hr = HybridRetriever(self.model)

    def answer(self, question: str):
        """一次完整问答。返回 (answer_text, sources, usage_dict)。

        sources = [{"doc": 文件名, "snippet": 片内容前40字}, ...]，按精排顺序。
        """
        # ① 检索：混合宽召回 + 精排 + 元数据过滤，一条龙（全部复用 core）
        hits, _fused = retrieve_reranked(question, self.model, self.cross, self.hr)

        # ② 拼 Prompt：四条规则 + [1]-[K] 资料块
        msgs = build_messages(question, hits)

        # ③ 生成：chat() 自带超时/重试/Token 统计（week1 地基）
        result = chat(msgs, temperature=0.1)

        # ④ 组装 sources：每条命中一项，doc 取来源文件名，snippet 取片段前 40 字
        sources = [
            {"doc": source, "snippet": text[:40].replace("\n", " ")}
            for source, text in hits
        ]

        # ⑤ 返回（usage 的键是 in/out，与 chat.py 约定一致）
        return result["text"], sources, result["usage"]

    def answer_stream(self, question: str):
        """流式问答生成器：yield ("token", 文本) / ("sources", 列表) / ("usage", dict)。

        与 answer() 共用同一套检索与 Prompt，差别只在出口：
        把「等全部生成完一次性返回」换成「边生成边吐」。
        """
        hits, _fused = retrieve_reranked(question, self.model, self.cross, self.hr)
        msgs = build_messages(question, hits)

        # stream_options 让最后一帧带回 usage；个别网关不支持时降级重试一次
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat", messages=msgs, temperature=0.1,
                stream=True, stream_options={"include_usage": True},
            )
        except Exception:
            stream = client.chat.completions.create(
                model="deepseek-chat", messages=msgs, temperature=0.1, stream=True,
            )

        parts, usage = [], {"in": 0, "out": 0}
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield "token", delta          # 逐字吐出（SSE 帧内容）
            u = getattr(chunk, "usage", None)
            if u:                                 # include_usage 生效时最后一帧带它
                usage = {"in": u.prompt_tokens, "out": u.completion_tokens}

        yield "sources", [{"doc": s, "snippet": t[:40].replace("\n", " ")} for s, t in hits]
        yield "usage", usage
