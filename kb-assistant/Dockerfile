# ---- 企业知识库 RAG + Agent 工作台 ----
# 基础镜像：Python 3.11 精简版
FROM python:3.11-slim

# 镜像构建参数
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# 健康检查需要 curl
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 依赖层（利用缓存：只有 requirements 变化才重装）
COPY requirements.txt .
# 先装 CPU 版 torch（避免 PyPI 默认 CUDA 版的 ~5GB 冗余，embedding 本来就在 CPU 跑）
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt -i ${PIP_INDEX}

# 代码层
COPY core/ ./core/
COPY app/ ./app/
COPY corpus/ ./corpus/
COPY data/ ./data/
COPY corpus_sources.json .

# 说明：模型缓存 / 向量库 / 切片产物 不打进镜像，运行时挂载（体积与更新都更友好）
#   ~/.cache/huggingface  -> /root/.cache/huggingface
#   ./chroma_db           -> /app/chroma_db
#   ./index               -> /app/index
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
