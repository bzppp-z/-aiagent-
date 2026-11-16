"""Day 36-38: 最小 Agent 循环 —— ReAct + 工具调用 + 步数上限 + Trace

链路位置: 用户问题 → [模型决定调什么工具] → [执行工具] → [结果回灌] → 循环 → 最终回答
用法: python core/agent.py            # 跑内置演示（检索 + 计算 + 组合）
      python core/agent.py "你的问题"

设计要点:
  1. 工具调用走 OpenAI function calling（week1 Day 6 的地基能力）
  2. 循环上限 MAX_STEPS，防止模型无限调工具烧钱
  3. 每步记录 Trace（思考/工具/参数/结果/耗时）——出问题能定位是哪一步
  4. 两个工具：search_kb 检索课程知识库；calculator 安全四则运算
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import ast
import json
import operator
import sys
import time

CORE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CORE)

from sentence_transformers import SentenceTransformer, CrossEncoder
from retrieval import HybridRetriever
from rag import retrieve_reranked, RERANKER
from embed import MODEL_NAME
from chat import client

MAX_STEPS = 5          # 循环上限：防止无限调用工具
TOOL_PREVIEW = 80      # Trace 里工具结果预览长度

SYSTEM = (
    "你是课程助教 Agent。你可以调用工具：\n"
    "- search_kb：检索课程知识库（回答课程内容、学习安排、项目问题时必须先调用它）\n"
    "- calculator：计算四则运算\n"
    "规则：\n"
    "1. 需要课程知识时，先调用 search_kb，再依据返回片段回答，并标注来源编号 [n]；\n"
    "2. 需要算术时调用 calculator，不要自己心算；\n"
    "3. 工具返回的资料不足以回答时，明确说「资料中没有相关内容」；\n"
    "4. 拿到足够信息后直接给出最终回答，不要反复调用同一个工具。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "检索课程知识库，返回与问题最相关的课程片段（带来源）",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索问题，尽量完整"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算四则运算表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 12*(3+4)"}},
                "required": ["expression"],
            },
        },
    },
]

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def safe_eval(expr: str) -> str:
    """AST 安全求值：只允许数字和四则运算，杜绝 eval 注入。"""
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("不支持的表达式")

    return str(_ev(ast.parse(expr, mode="eval").body))


class CourseAgent:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.cross = CrossEncoder(RERANKER, max_length=512)
        self.hr = HybridRetriever(self.model)

    # ---------- 工具实现 ----------
    def search_kb(self, query: str, k: int = 3) -> str:
        hits, _ = retrieve_reranked(query, self.model, self.cross, self.hr, k=k)
        parts = [f"[{i}] ({src})\n{text[:300]}" for i, (src, text) in enumerate(hits, 1)]
        return "\n\n".join(parts) if parts else "（知识库没有检索到相关片段）"

    def calculator(self, expression: str) -> str:
        try:
            return safe_eval(expression)
        except Exception as e:
            return f"计算失败：{e}"

    def dispatch(self, name: str, args: dict) -> str:
        if name == "search_kb":
            return self.search_kb(args.get("query", ""))
        if name == "calculator":
            return self.calculator(args.get("expression", ""))
        return f"未知工具：{name}"

    # ---------- 主循环 ----------
    def run(self, question: str, max_steps: int = MAX_STEPS):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        trace = []

        for step in range(1, max_steps + 1):
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                temperature=0.1,
                timeout=60,
            )
            msg = resp.choices[0].message
            cost_ms = int((time.perf_counter() - t0) * 1000)

            # 没有工具调用 → 最终回答
            if not msg.tool_calls:
                trace.append({"step": step, "type": "final", "cost_ms": cost_ms})
                return msg.content or "（模型未返回内容）", trace, messages

            # 有工具调用 → 逐个执行，结果回灌
            calls = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                t1 = time.perf_counter()
                result = self.dispatch(tc.function.name, args)
                tool_ms = int((time.perf_counter() - t1) * 1000)
                trace.append({
                    "step": step, "type": "tool",
                    "name": tc.function.name, "args": args,
                    "result_preview": result[:TOOL_PREVIEW].replace("\n", " "),
                    "cost_ms": cost_ms, "tool_ms": tool_ms,
                })
                calls.append({
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
            messages.insert(-len(calls), {"role": "assistant", "content": msg.content, "tool_calls": calls})

        trace.append({"step": max_steps, "type": "limit"})
        return "（达到最大步数上限，未能完成）", trace, messages


def show(question, answer, trace):
    print(f"\n{'=' * 60}\n[问题] {question}")
    print("-" * 60)
    for t in trace:
        if t["type"] == "tool":
            print(f"  步骤{t['step']} 调工具 {t['name']} {t['args']} → {t['result_preview']}… "
                  f"(LLM {t['cost_ms']}ms / 工具 {t['tool_ms']}ms)")
        elif t["type"] == "final":
            print(f"  步骤{t['step']} 生成最终回答 (LLM {t['cost_ms']}ms)")
        else:
            print(f"  步骤{t['step']} 达到上限")
    print("-" * 60)
    print(answer)


if __name__ == "__main__":
    agent = CourseAgent()
    questions = sys.argv[1:] or [
        "第 5 周要做什么？",
        "帮我算一下 128 * 37 等于多少",
        "课程里 TTFT 指什么？顺便算出 96 * 1000 分之一是多少",
    ]
    for q in questions:
        ans, tr, _ = agent.run(q)
        show(q, ans, tr)
