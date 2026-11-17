# 企业知识库 RAG + Agent 工作台

> 用自然语言问它课程相关的问题，它基于语料回答，**每个论点带 [n] 来源引用**；语料里没有的内容明确拒答，而不是编造。

一个边做边学的实战项目：文档入库 → 混合检索（向量 + BM25 → RRF 融合）→ Rerank 精排 → 带引用生成，外加 Agent 模式（ReAct + 工具调用）与量化评测闭环（Recall@5 + LLM 裁判），每个组件都用评测数据验证有没有用。

## 亮点

- **拒答门控**：语料外问题正确拒答，幻觉有数据（32 条黄金集，拒答 100% / 生成端忠实度 5.00）
- **检索优化闭环**：结构感知切分、元数据过滤、混合检索、查询改写、Rerank——每项优化都有前后对照指标（Recall@5 96.3%→100%，MRR 0.880→0.948）
- **服务化**：FastAPI + SSE 流式 + 缓存/并发/Token 预算三道可靠性闸门，Locust 压测定位瓶颈（CPU Cross-Encoder 精排）
- **容器化部署**：Docker 镜像 2.66GB（CPU 版 torch，省去 ~5GB CUDA 冗余），已上线阿里云 ECS

## 目录

完整代码在本仓库 `kb-assistant/` 子目录，详细文档见：

- [`kb-assistant/README.md`](./kb-assistant/README.md) — 架构 / 快速开始 / 容器化部署 / 路线图 / Bad Case 修复台账
- [`kb-assistant/docs/需求文档.md`](./kb-assistant/docs/需求文档.md) — 需求与候选优化池
- [`kb-assistant/docs/容量报告.md`](./kb-assistant/docs/容量报告.md) — 压测与容量规划
