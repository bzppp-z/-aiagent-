"""v0.3 / FR-3.4: 多轮对话 + 查询改写

链路位置: 追问 → [查询改写: 历史浓缩成完整问题] → 混合检索 → 精排 → 生成
用法:
  python core/rag_chat.py                  # 跑内置演示（含指代消解对照）
  python core/rag_chat.py "问1" "问2"      # 按顺序当一轮会话跑
  python core/rag_chat.py --live           # 交互模式，退出输 q

核心设计: 改写进检索、不进生成。
  追问"它怎么部署？"缺主语，直接检索必翻车；
  改写器（LLM 小调用）读最近几轮历史，把它补全成独立问题再检索；
  生成时只看改写后的问题——历史已经浓缩在里面，不用重复塞，省 Token。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
from sentence_transformers import SentenceTransformer
from chat import chat
from rag import retrieve, build_messages      # 复用 v0.1 的检索与 Prompt 拼装
from embed import MODEL_NAME

HISTORY_TURNS = 3     # 改写器最多回看几轮历史
ANSWER_SNIPPET = 200  # 历史里助教回答截断长度（控制改写调用的 Token）


REWRITE_SYSTEM = (
    "你是检索查询改写器。根据对话历史，把学员的最新追问改写成一个不依赖上下文、"
    "可以直接拿去知识库检索的完整问题。\n"
    "规则：\n"
    "1. 补全指代：「它/这个/上面说的」等要替换成历史里的具体对象；\n"
    "2. 只用对话历史里出现过的信息补全，不要引入新知识；\n"
    "3. 如果追问本身已经是完整问题，原样返回；\n"
    "4. 只输出改写后的问题本身，不要任何解释、引号或前缀。"
)


def rewrite_question(history, question):
    """查询改写：历史 + 追问 → 独立完整问题。历史为空（第一问）则跳过。"""
    if not history:
        return question, False          # 第一问没有指代可补，省一次调用
    hist_text = "\n".join(
        f"学员: {q}\n助教: {a[:ANSWER_SNIPPET]}" for q, a in history[-HISTORY_TURNS:]
    )
    msgs = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"【对话历史】\n{hist_text}\n\n【最新追问】{question}"},
    ]
    rewritten = chat(msgs, temperature=0)["text"].strip()
    return rewritten, True


def answer(question, history, model):
    """一轮完整问答: 改写 → 检索 → 生成。返回 (改写后问题, chat结果)。"""
    standalone, did_rewrite = rewrite_question(history, question)
    hits = retrieve(standalone, model)               # 用改写后的问题检索
    msgs = build_messages(standalone, hits)          # 生成也基于改写后的问题
    result = chat(msgs, temperature=0.1)
    return standalone, did_rewrite, result


def show_turn(no, raw_q, standalone, did_rewrite, result, model, contrast=False):
    print(f"\n{'=' * 60}\n[第 {no} 问] {raw_q}")
    if did_rewrite:
        print(f"[改写为] {standalone}")
    print(f"{'-' * 60}")
    print(result["text"])
    print("-" * 60)
    print(f"Token: 输入 {result['usage']['in']} / 输出 {result['usage']['out']}")
    if contrast and did_rewrite:
        # 对照: 若不做改写、拿原始追问直接检索, Top-1 会是什么？（展示改写的价值）
        src, text = retrieve(raw_q, model, k=1)[0]
        print(f"[对照] 不改写直接检索『{raw_q}』→ Top-1: [{src}] {text[:34]}...")


def run(questions, live=False):
    model = SentenceTransformer(MODEL_NAME)
    history = []
    no = 0
    if live:
        print("多轮会话（输 q 退出）")
        while True:
            q = input("\n学员: ").strip()
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            no += 1
            standalone, did, result = answer(q, history, model)
            show_turn(no, q, standalone, did, result, model)
            history.append((q, result["text"]))
    else:
        for q in questions:
            no += 1
            standalone, did, result = answer(q, history, model)
            show_turn(no, q, standalone, did, result, model, contrast=True)
            history.append((q, result["text"]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--live":
        run([], live=True)
    elif args:
        run(args)
    else:
        # 演示: 选型 → 指代追问 → 指代追问, 对照列展示"不改写会怎样"
        run([
            "Embedding 模型怎么选？",
            "它怎么部署？",            # 「它」= 上一轮的 Embedding 模型
            "部署成本大概多少？",        # 承接上一轮的「部署」话题
        ])
