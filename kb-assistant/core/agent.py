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

STATE_THINK, STATE_ACT, STATE_DONE, STATE_LIMIT = "THINK", "ACT", "DONE", "LIMIT"

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
            "name": "list_files",
            "description": "列出知识库中的文档清单（文件名与切片数），用于回答「知识库有哪些资料 / 覆盖什么主题」这类问题",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_progress",
            "description": "查询学员当前学习进度：第几周第几天、已完成阶段、下一步计划。用于回答「我学到哪了 / 接下来学什么」",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
    def __init__(self, model=None, cross=None, hr=None):
        # 服务层可传入已加载的模型复用，避免重复占内存
        self.model = model or SentenceTransformer(MODEL_NAME)
        self.cross = cross or CrossEncoder(RERANKER, max_length=512)
        self.hr = hr or HybridRetriever(self.model)

    # ---------- 工具实现 ----------
    def search_kb(self, query: str, k: int = 3) -> str:
        hits, _ = retrieve_reranked(query, self.model, self.cross, self.hr, k=k)
        parts = [f"[{i}] ({src})\n{text[:300]}" for i, (src, text) in enumerate(hits, 1)]
        return "\n\n".join(parts) if parts else "（知识库没有检索到相关片段）"

    def list_files(self) -> str:
        """列知识库文档：按来源统计切片数（日志类文档单独说明，不参与检索）。"""
        from collections import Counter
        course, log = Counter(), Counter()
        for m in self.hr.meta:
            (log if m.get("kind") == "log" else course)[m["source"]] += 1
        lines = [f"- {name}（{n} 片）" for name, n in sorted(course.items())]
        text = f"知识库共 {len(course)} 篇文档、{sum(course.values())} 片切片：" + chr(10) + chr(10).join(lines)
        if log:
            text += f"{chr(10)}另有日志类文档 {len(log)} 篇（不参与检索）。"
        return text

    def get_progress(self) -> str:
        """读 data/progress.json，返回当前进度摘要。"""
        import json
        path = os.path.join(os.path.dirname(CORE), "data", "progress.json")
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            return f"（读取进度失败：{e}）"
        cur = d.get("current", {})
        lines = [
            f"当前：第 {cur.get('week')} 周 Day {cur.get('day')} —— {cur.get('stage')}",
            "已完成：",
        ]
        for c in d.get("completed", []):
            lines.append(f"- 第 {c['week']} 周（Day {c['days']}）：{c['topic']}")
        lines.append(f"下一步：{d.get('next')}")
        return chr(10).join(lines)

    def calculator(self, expression: str) -> str:
        try:
            return safe_eval(expression)
        except Exception as e:
            return f"计算失败：{e}"

    def dispatch(self, name: str, args: dict) -> str:
        if name == "search_kb":
            return self.search_kb(args.get("query", ""))
        if name == "list_files":
            return self.list_files()
        if name == "get_progress":
            return self.get_progress()
        if name == "calculator":
            return self.calculator(args.get("expression", ""))
        return f"未知工具：{name}"

    # ---------- 状态机主循环 ----------
    # 状态转移表：
    #   THINK --(无工具调用)--> DONE
    #   THINK --(有工具调用)--> ACT --(工具执行完)--> THINK
    #   THINK --(步数耗尽)----> LIMIT
    def run(self, question: str, max_steps: int = MAX_STEPS):
        state = {
            "node": STATE_THINK,
            "step": 0,
            "answer": None,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
        }
        trace = []

        while state["node"] in (STATE_THINK, STATE_ACT):
            if state["node"] == STATE_THINK:
                if state["step"] >= max_steps:          # 转移：THINK -> LIMIT
                    state["node"] = STATE_LIMIT
                    break
                state["step"] += 1

                t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=state["messages"],
                    tools=TOOLS,
                    temperature=0.1,
                    timeout=60,
                )
                msg = resp.choices[0].message
                cost_ms = int((time.perf_counter() - t0) * 1000)

                if not msg.tool_calls:                  # 转移：THINK -> DONE
                    state["node"] = STATE_DONE
                    state["answer"] = msg.content or "（模型未返回内容）"
                    trace.append({"step": state["step"], "type": "final", "cost_ms": cost_ms})
                    break

                state["pending_calls"] = msg                  # 转移：THINK -> ACT
                state["node"] = STATE_ACT
                state["llm_cost_ms"] = cost_ms
                state["assistant_content"] = msg.content

            elif state["node"] == STATE_ACT:
                msg = state["pending_calls"]
                calls = []
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    t1 = time.perf_counter()
                    result = self.dispatch(tc.function.name, args)
                    tool_ms = int((time.perf_counter() - t1) * 1000)
                    trace.append({
                        "step": state["step"], "type": "tool",
                        "name": tc.function.name, "args": args,
                        "result_preview": result[:TOOL_PREVIEW].replace(chr(10), " "),
                        "cost_ms": state["llm_cost_ms"], "tool_ms": tool_ms,
                    })
                    calls.append({
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    })
                    state["messages"].append({"role": "tool", "tool_call_id": tc.id, "content": result})
                state["messages"].insert(
                    -len(calls),
                    {"role": "assistant", "content": state["assistant_content"], "tool_calls": calls},
                )
                state["node"] = STATE_THINK                 # 转移：ACT -> THINK

        if state["node"] == STATE_LIMIT:
            trace.append({"step": max_steps, "type": "limit"})
            return "（达到最大步数上限，未能完成）", trace, state["messages"]

        return state["answer"], trace, state["messages"]


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
