"""Day 33: locust 压测 —— 非流式问答（P95/P99/吞吐）+ SSE 首字时间（TTFT）

用法（项目根目录）:
  先起服务（压测时放宽预算与并发）:
    DAILY_TOKEN_BUDGET=5000000 MAX_CONCURRENT=8 .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
  再压测:
    .venv/Scripts/python.exe -m locust -f loadtest/locustfile.py --headless \
        -u 4 -r 1 -t 60s --host http://127.0.0.1:8000 --csv loadtest/result --only-summary

设计说明:
  - 每个问题加随机后缀做缓存击穿，保证压的是真实链路（检索+精排+生成），不是缓存；
  - 429（并发闸门拒绝）计为成功但单独记录——容量规划要看"拒绝率"这个信号；
  - TTFT 用 SSE 第一帧到达时间近似（requests 流式读首行），单独作为一个统计项上报。
"""
import json
import random
import time
import uuid

from locust import HttpUser, between, events, task

QUESTIONS = [
    "第六周学什么？",
    "切片大小一般设多少合适？",
    "Embedding 和 Rerank 有什么区别？",
    "为什么离线在线必须用同一个 Embedding 模型？",
    "chunk 大小设为 64 token 是文中的什么例子？",
]


class QAUser(HttpUser):
    wait_time = between(1, 2)

    def unique(self, q: str) -> str:
        # 加随机后缀击穿缓存：压测要量真实链路成本
        return f"{q}（压测{uuid.uuid4().hex[:4]}）"

    @task(3)
    def ask(self):
        q = self.unique(random.choice(QUESTIONS))
        with self.client.post("/api/chat", json={"question": q},
                              name="/api/chat", catch_response=True) as r:
            if r.status_code == 429:
                r.success()          # 限流是预期行为；拒绝数在 /api/stats 里单看
            elif r.status_code != 200:
                r.failure(f"status={r.status_code}")

    @task(1)
    def ask_stream_ttft(self):
        q = self.unique(random.choice(QUESTIONS))
        t0 = time.perf_counter()
        with self.client.get("/api/chat/stream", params={"question": q},
                             stream=True, name="/api/chat/stream", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status={r.status_code}")
                return
            try:
                first = next(r.iter_lines(), b"")     # 等待第一帧（首字）
                ttft_ms = (time.perf_counter() - t0) * 1000
                events.request.fire(
                    request_type="SSE", name="TTFT(ms)",
                    response_time=ttft_ms, response_length=len(first or b""),
                    exception=None, context={},
                )
                r.success()
            except StopIteration:
                r.failure("empty stream")
