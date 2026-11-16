"""v0.5 FastAPI 服务入口

运行（项目根目录）:
  .venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
  → 接口文档自动生成在 http://localhost:8000/docs
  → 测试: curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"question\":\"第六周学什么\"}"
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import ChatRequest, ChatResponse, Source, Usage
from app.session import SessionStore
from app.service import RAGService

sessions = SessionStore()
service: RAGService | None = None


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
    """非流式问答：POST {question, session_id?} → {answer, sources, usage, session_id, latency_ms}"""
    if service is None:
        raise HTTPException(503, "服务尚未就绪")

    t0 = time.perf_counter()
    sid = req.session_id or sessions.new_id()

    # ① 调服务层：检索 + 拼 Prompt + 生成（耗时主体；def 路由由 FastAPI 丢线程池）
    answer_text, sources, usage = service.answer(req.question)

    # ② 写会话：同一 session_id 延续对话（供前端气泡展示 / 将来接查询改写）
    sessions.append(sid, req.question, answer_text)

    # ③ 组装响应——Pydantic 模型兜住出参形状，字段缺漏会在返回时立刻报错
    return ChatResponse(
        answer=answer_text,
        sources=[Source(**s) for s in sources],
        usage=Usage(input_tokens=usage["in"], output_tokens=usage["out"]),
        session_id=sid,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# TODO(Day 29-30 下半场): 流式接口
#   @app.get("/api/chat/stream")
#   async def chat_stream(question: str, session_id: str | None = None):
#       用 sse_starlette.EventSourceResponse，把 openai stream=True 的 chunk 逐帧 yield
