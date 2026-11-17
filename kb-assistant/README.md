# kb-assistant — 企业知识库 RAG + Agent 工作台

> 企业知识库 RAG + Agent 工作台（带评测闭环）：文档入库、混合检索、精排、带引用生成，外加 Agent 模式（模型自主调用工具）与量化评测。
> **边做项目边学**——每个新学的能力作为一个组件装入，并用评测数据验证它有没有用。

## 它能做什么

用自然语言问它课程相关的问题，它基于课程语料回答，**每个论点带 [n] 来源引用**；语料里没有的内容会明确拒答，而不是编造。

```
你: 切片大小一般设多少合适？
它: 按场景分三档——对话问答 256~512 token，知识库检索 512~1024 token，
    长文档摘要 1500~2000 token，需按业务效果调整 [1]。
    （引用: how_to_chunking.md）
```

## 架构

```
【离线 · 建库一次】                          【在线 · 每问都走】
corpus_sources.json 登记语料源               用户提问
   ↓ 清洗(剥元信息)                          ↓ 查询改写 (v0.3)
   ↓ 结构感知切分(按标题, 500字兜底)          ↓ 混合检索: 向量+BM25 → RRF 融合
   ↓ bge-small-zh 向量化(512维)             ↓ Rerank 精排 top-5 (v0.2)
   ↓ ChromaDB 入库(cosine)                  ↓ 拼 Prompt(带[n]编号+来源)
                                            ↓ DeepSeek 生成带引用回答
【评测闭环 v0.4】评测集 → Recall@5 + LLM裁判 → 优化有据可依
```

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .env 里配置 DEEPSEEK_API_KEY（见 .env.example）

# 建库（离线，语料更新后重跑）
python core/chunking.py struct                  # 清洗+结构切分 → index/chunks_struct/
python core/embed.py index/chunks_struct        # 向量化 → vectors.npy + meta.json
python core/rag_db.py                           # 入 ChromaDB（集合 kb_ta_struct）

# 问答（在线）
python core/rag.py "第六周的学习内容是什么？"
python core/rag.py                              # 默认验收三问
python core/rag_chat.py --live                  # 多轮会话（v0.3）
```

## 容器化部署

> 模型缓存、向量库、切片产物不打进镜像，运行时挂载——镜像只有代码和依赖。

```bash
# 构建（首次约 10-15 分钟；Dockerfile 先装 CPU 版 torch，避免 PyPI 默认拉下 ~5GB CUDA 全家桶）
docker build -t kb-assistant:0.6 .

# 运行（挂载模型缓存 + 知识库产物 + .env）
docker run -d --name kb-assistant -p 8000:8000   --env-file .env   -e HF_HUB_OFFLINE=1   -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro"   -v "$(pwd)/chroma_db:/app/chroma_db"   -v "$(pwd)/index:/app/index"   -v "$(pwd)/logs:/app/logs"   kb-assistant:0.6

# 或一键
docker compose up -d --build
```

启动后访问 http://localhost:8000/ （网页）或 /docs （接口文档）。
注意：知识库（chroma_db / index）需先在宿主机建好（见「快速开始」的建库三步）。

### 实机验证记录（Windows 本机 + 阿里云 ECS）

镜像 kb-assistant:0.6，**2.66GB**——Dockerfile 先装 CPU 版 torch 占位，pip 跳过 CUDA 版（省去 nvidia 库 3GB + triton 1GB 的纯冗余，首次构建 10.1GB→2.66GB）。启动约 60 秒（加载本地模型）。

```bash
# 1) 拉不到 Docker Hub 时：用国内镜像站拉基础镜像并打标签
docker pull docker.1ms.run/library/python:3.11-slim
docker tag docker.1ms.run/library/python:3.11-slim python:3.11-slim

# 2) 构建
docker build -t kb-assistant:0.6 .

# 3) 运行（挂载路径用 Windows 风格 C:/ 前缀）
docker run -d --name kb-assistant -p 8000:8000 \
  --env-file .env -e HF_HUB_OFFLINE=1 \
  -v "C:/Users/bzp20/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "C:/Users/bzp20/Desktop/ai学习/my-ai-projects/kb-assistant/chroma_db:/app/chroma_db" \
  -v "C:/Users/bzp20/Desktop/ai学习/my-ai-projects/kb-assistant/index:/app/index" \
  -v "C:/Users/bzp20/Desktop/ai学习/my-ai-projects/kb-assistant/logs:/app/logs" \
  kb-assistant:0.6

curl http://localhost:8000/health   # status ok / model_ready true
```

**六个实战坑**（都踩过并修复）：
1. requirements.txt 漏了 fastapi/uvicorn/sse-starlette——本地 venv 手动装过所以没暴露，容器从零安装立刻报 ModuleNotFoundError；
2. Docker Hub 域名解析失败（lookup auth.docker.io: no such host）——用 docker.1ms.run 镜像站拉基础镜像再 tag；
3. 挂载路径写成 /c/... 会变成空目录——必须用 Windows 风格 C:/...；
4. docker exec ... ls /app/... 被 Git Bash 路径转换劫持——加 MSYS_NO_PATHCONV=1；
5. PyPI 默认 torch 是 GPU 完整版——sentence-transformers 会连带拉下 ~5GB CUDA 依赖，必须在装 requirements **之前**用 `pip install torch --index-url https://download.pytorch.org/whl/cpu` 先占位（pip 见 torch 已满足就跳过）；
6. 阿里云机房直连 huggingface.co 不通——模型先用 `HF_ENDPOINT=https://hf-mirror.com` 下载进挂载目录，容器保持 HF_HUB_OFFLINE=1 离线运行。

### 阿里云 ECS 部署记录

配置：2 vCPU / 2GiB（+2G swap）/ 40G 盘，Ubuntu 22.04。

- 镜像分发不走仓库：`docker save | gzip` 导出 539MB，scp 上服务器 `docker load`（自动还原 2.66GB）
- 模型：hf-mirror 下载进 `/opt/kb/hf-cache` 挂载（注意 git clone 下来的模型目录结构与 HF cache 不兼容，不能直接用）
- 实测：`/health` ok、model_ready true；真实问答全链路（改写→检索→精排→生成）18.3s——瓶颈在 2 核 CPU 跑 Cross-Encoder 精排（约 6-9s），与压测结论一致；规模化后迁 GPU（见 docs/容量报告.md 的 ROI 排序）

## 语料

知识来源由 `corpus_sources.json` 配置化接入：

- **自有材料**（随仓库上传）：学习计划、教学笔记、Bad Case 台账、README
- **第三方教程**（llm-master 仓库，只登记路径不入库）：约 40 篇 LLM/RAG 专题文章

语料里没有的知识，助教会明确说"资料中没有相关内容"——这是设计出来的行为，不是缺陷（对抗幻觉的第一道闸门）。

## 路线图（详见 docs/需求文档.md）

| 版本 | 内容 | 状态 |
|---|---|---|
| v0.1 | 地基迁移 + 多源语料入库 + CLI 问答带引用/拒答 | ✅ 2026-11-14 |
| v0.2 | 混合宽召回 top-20 + Rerank 精排（已接入 rag.py 主链路，RAG_K 可调） | ✅ 2026-11-14 |
| v0.3 | 多轮对话 + 查询改写（`rag_chat.py`，指代消解） | ✅ 2026-11-14 |
| v0.4 | **评测闭环**：评测集 + Recall@5 + LLM 裁判 | ✅ 2026-11-15 |
| v0.5 | FastAPI 工作台 + SSE 流式 + 压测 | ✅ 2026-11-16 |
| v0.6 | Agent 工作台：ReAct 状态机（THINK/ACT）+ 工具调用 + Trace | ✅ 2026-11-16 |
| 部署 | Docker 容器化（镜像瘦身 10.1GB→2.66GB）+ 阿里云 ECS 上线 | ✅ 2026-11-17 |

## Bad Case 修复台账（节选）

| 案例 | 现象 | 定位 | 修复 | 效果 |
|---|---|---|---|---|
| 切片大小经验值 | 答"资料中没有"，但语料有 | 检索侧：元信息块污染 + 固定长度切分焊接语义 | 清洗 + 结构感知切分 | 目标段排名 12→1 |
| overlap 设 0 | 召回正常但过度保守拒答 | 生成侧：拒答规则过于绝对 | Prompt 加"允许一步推断" | 正确推理，且语料外问题仍拒答 |

完整记录见 `corpus/bad_cases.md`——它是评测闭环（v0.4）的原料。
