# kb-assistant — 课程助教

> 一个带评测闭环的知识库 RAG 项目：把六周 LLM/RAG 课程学到的技术装进一个自己天天用的助手。
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

## 语料

知识来源由 `corpus_sources.json` 配置化接入：

- **自有材料**（随仓库上传）：学习计划、教学笔记、Bad Case 台账、README
- **第三方教程**（llm-master 仓库，只登记路径不入库）：约 40 篇 LLM/RAG 专题文章

语料里没有的知识，助教会明确说"资料中没有相关内容"——这是设计出来的行为，不是缺陷（对抗幻觉的第一道闸门）。

## 路线图（详见 docs/需求文档.md）

| 版本 | 内容 | 状态 |
|---|---|---|
| v0.1 | 地基迁移 + 多源语料入库 + CLI 问答带引用/拒答 | 🚧 进行中 |
| v0.2 | 混合宽召回 top-20 + Rerank 精排 | ⏳ |
| v0.3 | 多轮对话 + 查询改写（`rag_chat.py`，指代消解） | ✅ 2026-11-14 |
| v0.4 | **评测闭环**：评测集 + Recall@5 + LLM 裁判 | ⏳ |
| v0.5 | FastAPI 工作台 + SSE 流式 + 压测 | ⏳ |
| v0.6 | Agent 化：查进度工具 + ReAct 循环 | ⏳ |

## Bad Case 修复台账（节选）

| 案例 | 现象 | 定位 | 修复 | 效果 |
|---|---|---|---|---|
| 切片大小经验值 | 答"资料中没有"，但语料有 | 检索侧：元信息块污染 + 固定长度切分焊接语义 | 清洗 + 结构感知切分 | 目标段排名 12→1 |
| overlap 设 0 | 召回正常但过度保守拒答 | 生成侧：拒答规则过于绝对 | Prompt 加"允许一步推断" | 正确推理，且语料外问题仍拒答 |

完整记录见 `corpus/bad_cases.md`——它是评测闭环（v0.4）的原料。
