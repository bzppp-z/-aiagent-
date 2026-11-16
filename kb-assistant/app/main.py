"""v0.5 FastAPI 服务入口

运行（项目根目录）:
  .venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
  → 接口文档自动生成在 http://localhost:8000/docs
  → 测试: curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"question\":\"第六周学什么\"}"
"""
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from app.reliability import ConcurrencyLimiter, SimpleCache, TokenBudget
from app.schemas import ChatRequest, ChatResponse, Source, Usage
from app.session import SessionStore
from app.service import RAGService

sessions = SessionStore()
service: RAGService | None = None

# —— Day 31-32：可靠性 + 成本组件（进程级单例）——
CACHE = SimpleCache(maxsize=128)                                    # 相同问题秒回
LIMITER = ConcurrencyLimiter(int(os.environ.get("MAX_CONCURRENT", "4")))   # 并发闸门
BUDGET = TokenBudget(int(os.environ.get("DAILY_TOKEN_BUDGET", "200000")))  # 日预算


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载重资源（模型/索引），服务期共享——绝不在请求里加载。"""
    global service
    service = RAGService()
    print("[启动] RAGService 就绪")
    yield


app = FastAPI(title="kb-assistant 课程助教", version="0.5", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": service is not None}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """非流式问答 —— 三道关：预算熔断 → 缓存命中 → 并发闸门。"""
    if service is None:
        raise HTTPException(503, "服务尚未就绪")
    if not BUDGET.check():                                  # 关 ①：日预算熔断
        raise HTTPException(429, "今日 Token 预算已耗尽，请明天再试")

    t0 = time.perf_counter()
    sid = req.session_id or sessions.new_id()

    hit = CACHE.get(req.question)                           # 关 ②：缓存命中秒回
    if hit:
        answer_text, sources, usage = hit
        sessions.append(sid, req.question, answer_text)
        return ChatResponse(
            answer=answer_text,
            sources=[Source(**s) for s in sources],
            usage=Usage(input_tokens=usage["in"], output_tokens=usage["out"]),
            session_id=sid,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            cached=True,
        )

    if not LIMITER.try_acquire():                           # 关 ③：并发满了立即 429
        raise HTTPException(429, "服务繁忙（并发已满），请稍后重试")
    try:
        answer_text, sources, usage = service.answer(req.question)
    finally:
        LIMITER.release()

    BUDGET.add(usage)                                       # 记账 + 回填缓存
    CACHE.set(req.question, (answer_text, sources, usage))
    sessions.append(sid, req.question, answer_text)

    return ChatResponse(
        answer=answer_text,
        sources=[Source(**s) for s in sources],
        usage=Usage(input_tokens=usage["in"], output_tokens=usage["out"]),
        session_id=sid,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/api/chat/stream")
async def chat_stream(question: str, session_id: str | None = None):
    """SSE 流式问答：event:token 逐字 → event:sources → event:usage → event:done

    设计要点：
    - 查询参数走 GET（EventSource 只能 GET），question 放 ?question=...；
    - service.answer_stream 是同步阻塞生成器 → iterate_in_threadpool 逐块转异步，
      绝不阻塞事件循环（概念 3 那个坑的解法）；
    - 帧格式交给 sse-starlette，它负责 data:/event:/心跳。
    """
    if service is None:
        raise HTTPException(503, "服务尚未就绪")
    if not BUDGET.check():
        raise HTTPException(429, "今日 Token 预算已耗尽，请明天再试")
    sid = session_id or sessions.new_id()

    hit = CACHE.get(question)                               # 缓存命中：全文一帧吐出，前端无感
    if hit:
        answer_text, hit_sources, usage = hit
        sessions.append(sid, question, answer_text)

        async def cached_gen():
            yield {"event": "token", "data": json.dumps({"t": answer_text}, ensure_ascii=False)}
            yield {"event": "sources", "data": json.dumps(hit_sources, ensure_ascii=False)}
            yield {"event": "usage", "data": json.dumps(usage)}
            yield {"event": "done", "data": json.dumps({"session_id": sid})}

        return EventSourceResponse(cached_gen())

    if not LIMITER.try_acquire():
        raise HTTPException(429, "服务繁忙（并发已满），请稍后重试")

    def gen():
        parts, srcs, usage = [], [], {"in": 0, "out": 0}
        try:
            for kind, payload in service.answer_stream(question):
                if kind == "token":
                    parts.append(payload)
                    yield {"event": "token", "data": json.dumps({"t": payload}, ensure_ascii=False)}
                elif kind == "sources":
                    srcs = payload
                    yield {"event": "sources", "data": json.dumps(payload, ensure_ascii=False)}
                elif kind == "usage":
                    usage = payload
                    yield {"event": "usage", "data": json.dumps(payload)}
            BUDGET.add(usage)
            CACHE.set(question, ("".join(parts), srcs, usage))
            sessions.append(sid, question, "".join(parts))
            yield {"event": "done", "data": json.dumps({"session_id": sid})}
        finally:
            LIMITER.release()                               # 无论正常结束还是客户端断开都释放名额

    return EventSourceResponse(iterate_in_threadpool(gen()))


@app.get("/api/stats")
def stats():
    """运维看板：预算账单 / 缓存命中 / 并发拒绝数。"""
    return {
        "budget": BUDGET.stats(),
        "cache": {"hits": CACHE.hits, "misses": CACHE.misses, "maxsize": CACHE.maxsize},
        "limiter": {"limit": LIMITER.limit, "rejected": LIMITER.rejected},
    }


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    """产品级单页：多轮会话 + Markdown 渲染 + 来源卡片 + 流式打字。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html")
