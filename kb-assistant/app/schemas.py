"""接口契约：请求/响应模型（Pydantic 校验，免写校验代码）

规矩：接口收发的每个字段都在这里定义，main.py 只引用不裸传 dict。
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="学员问题")
    session_id: str | None = Field(None, description="会话 ID；不传则服务端新建")


class Source(BaseModel):
    doc: str = Field(..., description="来源文件名")
    snippet: str = Field(..., description="片段前 40 字预览")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    usage: Usage
    session_id: str
    latency_ms: int
    cached: bool = False   # 命中缓存的回答（相同问题秒回）
