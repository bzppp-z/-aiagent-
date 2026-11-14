# 大模型应用开发学习计划（RAG 主线 · 6 周）

> 本计划基于本目录下 `llm-master` 仓库（卡码笔记·大模型全栈学习路线）定制，
> 为「有 Python3 + 后端基础、无 AI 基础、全职学习」的学习者设计。
> 目标：**6 周做出一个可评估、可上线的知识库问答（RAG）项目**。

---

## 一、这份计划怎么用

**你的画像与目标：**

| 项目 | 内容 |
|---|---|
| 基础 | Python3 语法 + 后端基础，无 AI 基础 |
| 时间 | 全职，每天 4 小时以上 |
| 主线 | 知识库问答（RAG）项目 |
| 原则 | 轻原理、重项目（Transformer 数学、微调暂不学） |
| 模型接入 | 国内大模型 API（DeepSeek / GLM / Qwen） |

**三条使用规则：**

1. **每天节奏固定**：上午/前半段读当天文章（约 1 小时），其余时间全部动手写代码。只读不写 = 没学。
2. **代码全部手敲**，不要复制粘贴教程代码。报错、排查、修复的过程就是学习本身。
3. **链接都是本仓库相对路径**，按住 Ctrl 点击即可跳转（VS Code / Typora 中）。

**整体节奏总览：**

| 周次 | 主题 | 产出物 |
|---|---|---|
| Day 0 | 环境与 API 准备 | 能成功调用一次大模型 API |
| 第 1 周 | 模型调用基本功 | 迷你项目：命令行 AI 助手 |
| 第 2 周 | RAG 第一条链路 | 能跑通的知识库问答 Demo |
| 第 3 周 | RAG 优化 | Bad Case 本 + 检索质量明显提升 |
| 第 4 周 | RAG 评估与收尾 | **主线项目：可评估的知识库问答系统** |
| 第 5 周 | 服务化上线 | 可访问的在线服务 + 压测数据 |
| 第 6 周 | Agent 进阶（选修） | 带 Agent 能力的进阶版项目 |

---

## 二、Day 0：学习前准备（约半天）

### 2.1 开通大模型 API（选一家即可，推荐 DeepSeek）

| 平台 | 特点 | 建议 |
|---|---|---|
| [DeepSeek 开放平台](https://platform.deepseek.com/) | 极便宜、OpenAI 兼容接口 | **首选**，充 10 元够学完整个计划 |
| 智谱 GLM 开放平台 | 有免费模型（glm-4-flash 等免费） | 备选 / 零成本方案 |
| 阿里百炼（Qwen） | 新用户有免费额度，模型选择多 | 备选 |

> 三家都兼容 OpenAI SDK：只需改 `base_url` 和 `api_key`，代码完全一样。
> 本仓库所有教程的示例代码都可以直接套用。

**注意**：DeepSeek **不提供** Embedding 接口，第 2 周做向量化的解决办法见 2.3。

### 2.2 开发环境

- Python 3.10+，用 `venv` 建独立虚拟环境
- IDE：VS Code 或 PyCharm
- 新建项目目录 `my-ai-projects/`（建议 `git init`，每次作业都提交）
- `pip` 换国内源（清华源），Windows 下建虚拟环境用 `python -m venv .venv`
- **安全习惯从第一天养成**：API Key 放 `.env` 文件里读取，并把 `.env` 写进 `.gitignore`——Key 泄漏到公开仓库会被盗刷

### 2.3 安装基础依赖（第 2 周的也一并装好）

```bash
pip install openai            # 调用大模型（所有国产 API 都用它）
pip install fastapi uvicorn   # 第 5 周服务化
pip install chromadb          # 第 2 周向量数据库（Windows 友好）
pip install sentence-transformers  # 第 2 周本地 Embedding 模型
```

> **Embedding 怎么选**：推荐直接用本地开源模型 `BAAI/bge-small-zh-v1.5`（免费、CPU 就能跑、中文效果好），代码里换模型名即可。不想装本地模型的话，用智谱或阿里百炼的 Embedding API。

### 2.4 开工前先读两篇建立全局认知（约 1 小时）

1. [大模型学习路线](llm-master/docs/llm/intro/llm_learning_roadmap.md) — 知道整个知识版图长什么样
2. [大模型应用开发到底在做什么](llm-master/docs/llm/intro/app_dev_overview.md) — 明确你未来 6 周练的就是这个

**Day 0 完成标志**：写一个 10 行的 Python 脚本，成功调用 DeepSeek（或 GLM/Qwen）并打印回答；终端里能看到消耗了多少 Token。

---

## 三、第 1 周：模型调用基本功

> 目标：从「会聊天」到「会开发」。这一周是一切的地基——后面 RAG、Agent
> 用到的流式、结构化输出、工具调用、成本控制全在这周学。

**本周文章（按顺序读）：**

1. [13 个大模型核心概念](llm-master/docs/llm/intro/llm_keywords.md)
2. [大模型如何训练](llm-master/docs/llm/intro/how_llm_trained.md)
3. [模型如何接入真实应用](llm-master/docs/llm/app/model_integration.md)
4. [Token、成本与延迟](llm-master/docs/llm/app/token_cost_latency.md)
5. [Prompt Engineering](llm-master/docs/llm/app/prompt_engineering.md)
6. [Few-shot、CoT 与 Reflection](llm-master/docs/llm/app/prompt_fewshot_cot_reflection.md)
7. [结构化输出](llm-master/docs/llm/app/structured_output.md)
8. [同步、异步与流式输出](llm-master/docs/llm/app/streaming_output.md)
9. [Function Calling](llm-master/docs/llm/app/function_calling.md)
10. [Context Engineering](llm-master/docs/llm/app/context_engineering.md)

**每日安排：**

| 天 | 上午读 | 下午动手 |
|---|---|---|
| Day 1 | ①② 概念扫盲 | 整理一张自己的「术语卡」（不懂的词查附录 A） |
| Day 2 | ③④ 接入与成本 | 封装一个 `chat.py` 模块：带超时、重试、错误处理、Token 统计 |
| Day 3 | ⑤⑥ 提示词 | 给助手写 System Prompt，做 few-shot 例子，对比提示词前后效果 |
| Day 4 | ⑦ 结构化输出 | 用 JSON Schema 让模型稳定输出结构化数据 + pydantic 校验 |
| Day 5 | ⑧ 流式输出 | 命令行打字机效果；再写一个 FastAPI 流式接口（你后端出身，这是热身） |
| Day 6 | ⑨ 工具调用 | 给助手加 2 个工具（如：查本地时间、简单计算器），走完 Function Calling 全流程 |
| Day 7 | ⑩ 上下文工程 | **迷你项目收尾**（见下） |

**迷你项目：命令行 AI 助手**

要求（对应仓库阶段项目标准）：
- 支持多轮对话、流式输出
- 支持 Function Calling（至少 2 个工具）
- 有一类输出走 JSON Schema 结构化（如「帮我记一条待办」）
- 有超时、重试、错误处理
- 每次对话后打印 Token 用量与累计成本（人民币）
- 用 `pytest` 写 3-5 个测试

**本周完成标志**：不看教程，能白板讲清「一次模型请求从发起到返回的完整链路」，以及 Token 如何计费。

---

## 四、第 2 周：RAG 第一条链路（先跑通）

> 目标：理解 RAG 解决什么问题，把「文档 → 切片 → 向量化 → 检索 → 生成」
> 五个环节逐一打通，拼出一个能用的 Demo。先跑通，不求好。

**本周文章（按顺序读）：**

1. [为什么需要 RAG](llm-master/docs/llm/app/why_rag.md)
2. [RAG 完整链路](llm-master/docs/llm/app/chain_of_rag.md)
3. [切片策略](llm-master/docs/llm/app/how_to_chunking.md)
4. [Embedding](llm-master/docs/llm/app/embedding.md)
5. [向量数据库](llm-master/docs/llm/app/vector_database.md)

**每日安排：**

| 天 | 读 | 动手 |
|---|---|---|
| Day 8 | ①② | 手画一张你的 RAG 链路图（贴在项目 README 里） |
| Day 9 | ③ | 文档加载模块：读几份 PDF/Markdown，实现固定长度+重叠切片；用不同 chunk 大小各切一版留着对比 |
| Day 10 | ④ | 用 bge-small-zh-v1.5 批量向量化切片；写个脚本算两条文本的相似度感受 Embedding 的含义 |
| Day 11 | ⑤ | ChromaDB 建库、入库、检索 Top-5，把检索结果打印出来人工看质量 |
| Day 12-13 | 复习本周 | **组装最小 RAG**：提问 → 检索 → 把片段塞进 Prompt → 生成带「来源引用」的回答 |
| Day 14 | — | 周复盘：写下 5 个你觉得回答得不好的例子（这就是下周的 Bad Case 本） |

**本周完成标志**：拿一份自己的文档（如某个软件说明书）提问，回答基本正确且带引用；故意问一个文档里没有的问题，观察它是否胡编（这个现象叫「幻觉」，第 3 周解决它）。

---

## 五、第 3 周：让 RAG 变好（优化）

> 目标：第 2 周的 Demo 一定有很多烂回答。这周系统学习排查与优化，
> 把 Bad Case 一个个消灭。**这周才是 RAG 的真功夫。**

**本周文章（按顺序读）：**

1. [RAG 常见问题排查](llm-master/docs/llm/app/rag_problems.md)
2. [RAG 优化](llm-master/docs/llm/app/rag_optimization.md)
3. [长文档与代码检索](llm-master/docs/llm/app/long_document_code_retrieval.md)

**每日安排：**

| 天 | 读 | 动手 |
|---|---|---|
| Day 15 | ① | 给上周的 Bad Case 分类：检索没召回？召回了没答对？还是模型瞎编？ |
| Day 16-17 | ②（上半） | 改进切片策略（按标题/语义切）；调 chunk 大小与 overlap，对比 Day 9 留下的版本 |
| Day 18 | ②（下半） | 加**混合检索**（BM25 关键词 + 向量），感受关键词兜底的价值 |
| Day 19 | ②（下半） | 加 **Rerank 重排**：本地 bge-reranker 或 API 均可；对比加与不加的 Top-3 质量 |
| Day 20 | ③ | 加**查询改写**与多轮对话支持（把「它怎么部署？」改写成完整问题） |
| Day 21 | — | 周复盘：每个 Bad Case 记录「现象 → 定位 → 改了什么 → 效果」 |

**本周完成标志**：Bad Case 本上至少 8 个案例有完整的分析和修复记录；同一个测试问题集，回答质量肉眼可见地变好。

---

## 六、第 4 周：让 RAG 可评估（主线项目收尾）

> 目标：「感觉变好了」不算数，要能拿出数字。这周建评测集、定指标、
> 做错误分析，把项目收成一个**简历级**的完整作品。

**本周文章：**

1. [RAG 评估](llm-master/docs/llm/app/rag_evaluation.md)（本周核心，精读）
2. [Agentic RAG](llm-master/docs/llm/intro/agentic_rag.md)（选读，开拓视野）

**每日安排：**

| 天 | 动手 |
|---|---|
| Day 22-23 | 建离线评测集：30-50 条「问题 + 标准答案 + 出处」，覆盖事实题、推理题、文档外题（考幻觉拒绝） |
| Day 24-25 | 实现自动评估：检索指标（召回命中率）+ 生成指标（用 LLM 当裁判评忠实度、相关性） |
| Day 26 | 错误分析：跑全量评测，把错题分类统计，找出当前系统最大短板 |
| Day 27 | 针对最大短板做最后一轮优化，记录优化前后指标对比 |
| Day 28 | **项目收尾**：README（架构图、技术选型理由、指标表）、demo 录屏、代码整理提交 git |

**主线项目完成标志**（对应仓库「企业知识库」项目标准）：
- ✅ 有混合检索、答案引用
- ✅ 有固定评测集（30+ 条）
- ✅ 有 Bad Case 分类和指标基线
- ✅ 能回答：为什么这样选型、效果如何度量、失败如何恢复

---

## 七、第 5 周：服务化上线

> 目标：把本地脚本变成别人能访问的服务。你有后端基础，这周会很快，
> 重点补 AI 服务特有的工程问题：流式、Token 成本、超时重试、压测。

**本周文章：**

1. [大模型部署方案选型](llm-master/docs/llm/app/deployment_options.md)（读「API 调用 vs 自部署」部分即可）
2. [服务压测与容量规划](llm-master/docs/llm/app/stress_testing.md)
3. [Token、成本与延迟](llm-master/docs/llm/app/token_cost_latency.md)（重读，这次带着服务视角）

> KV Cache、PagedAttention、量化这几篇讲的是**自己部署开源模型**才需要的知识，
> 你目前用 API，标注为「以后回来读」，本周跳过。

**每日安排：**

| 天 | 动手 |
|---|---|
| Day 29-30 | FastAPI 服务化：问答接口 + 流式接口（SSE）、会话管理、统一错误处理、请求日志 |
| Day 31 | 可靠性：超时、重试（指数退避）、并发限流、简单缓存（相同问题直接返回） |
| Day 32 | 成本控制：按用户/按天的 Token 统计与预算上限，超限熔断 |
| Day 33 | 压测：用 locust 压出 P95/P99 延迟、TTFT（首字时间）、最大吞吐；写一页容量报告 |
| Day 34-35 | 部署：Docker 打包，部署到一台云服务器（或本机 Docker Desktop）+ 周复盘 |

**本周完成标志**：把服务地址发给朋友，对方能在网页/接口上稳定提问并看到流式回答；有一份含 P99、TTFT、成本估算的压测报告。

---

## 八、第 6 周（选修）：Agent 进阶

> 主线已达成。这周把知识库升级成「会自己动手的助手」，也是通往
> Agent 工程方向的门。时间紧张可只读前 4 篇 + 做一个极简 Agent。

**本周文章（按顺序读）：**

1. [Agent 到底是什么](llm-master/docs/llm/app/agent_intro.md)
2. [Agent vs Workflow](llm-master/docs/llm/app/agent_vs_workflow.md)
3. [Agent 工具设计](llm-master/docs/llm/app/agent_tool_design.md)
4. [ReAct、Reflection 与规划执行](llm-master/docs/llm/app/react_reflection_planning.md)
5. [Agent 记忆](llm-master/docs/llm/app/agent_memory.md)（选读）
6. [Agent 失败模式](llm-master/docs/llm/app/agent_failure_modes.md)（选读）
7. [MCP 协议](llm-master/docs/llm/app/mcp_protocol.md)（选读）

**动手项目（二选一）：**

- **A. 给知识库加 Agent 能力**：除了检索工具，再加 1-2 个工具（计算器、联网搜索、查数据库），让模型自己决定调用什么——体会 RAG 与 Agent 的分界
- **B. 极简工具型 Agent**：一个 ReAct 循环 + 3 个工具 + 最大步数上限 + 简单 Trace 日志

**完成标志**：能说清「什么时候该用 Workflow、什么时候才需要 Agent」；Agent 失败时能看到 Trace 并定位是哪一步出的问题。

---

## 九、6 周之后学什么

| 方向 | 入口 | 什么时候学 |
|---|---|---|
| 求职面试 | [面试路线](llm-master/docs/roadmap/interview.md) | 想找工作时，用统一框架重新组织项目表达 |
| Transformer 原理 | [Transformer 专题](llm-master/docs/topics/transformer.md) | 想理解底层时，从直觉篇开始，不必啃数学 |
| 微调（SFT/LoRA） | [微调专题](llm-master/docs/topics/finetuning.md) | 当你遇到「Prompt 怎么写都调不好」的任务时 |
| AI 编程提效 | [AI 编程专题](llm-master/docs/topics/ai-coding.md) | 随时可学，用 Claude Code/Cursor 加速你写代码 |

---

## 十、遇到问题怎么办

1. **仓库的两种用法**：
   - 按计划顺序读（本计划已排好）；
   - 当字典查：出问题时去 [专题索引](llm-master/docs/topics/README.md) 找对应文章，如检索质量差 → [RAG 优化](llm-master/docs/llm/app/rag_optimization.md)。
2. **卡壳超过 1 小时**：先重读当天文章的对应小节，再带着具体报错/现象去搜，还解决不了就先绕过（记到问题清单里继续推进，很多问题几天后自然就懂了）。
3. **每周末写复盘**（半小时）：这周做了什么、哪个概念还是模糊、下周注意什么。六个复盘写完，你会看到自己清晰的成长轨迹。

---

## 附录 A：术语速查表（无 AI 基础必读）

| 术语 | 一句话解释 |
|---|---|
| LLM（大语言模型） | 读过海量文本后学会「预测下一个词」的模型，对话、写作、写代码都靠这个能力 |
| Token | 模型处理文本的最小单位，约等于 0.5-1.5 个汉字；**API 按 Token 收费** |
| 上下文窗口 | 模型一次能「看见」的文本上限（如 128K Token），超了就必须裁剪或检索 |
| Prompt | 你发给模型的输入。写 Prompt 的水平直接决定输出质量 |
| System Prompt | 设定模型角色和规则的提示词，用户看不见 |
| Temperature | 随机度旋钮：0 最稳定（适合抽取/结构化），调高更有创造性 |
| Embedding（嵌入） | 把文本变成一串数字（向量），意思相近的文本向量距离也近——机器理解语义的方式 |
| 向量数据库 | 专门存 Embedding 并快速找「最相似」的数据库，本计划用 ChromaDB |
| 切片（Chunking） | 把长文档切成小段再向量化。切得太碎丢上下文，太大检索不准——RAG 第一个坑 |
| RAG | 检索增强生成：先从你的资料里「查」，再让模型对着查到的内容「答」，解决模型不懂私有知识、爱编造的问题 |
| 幻觉 | 模型一本正经地编造事实。RAG + 引用 + 评测是主要对抗手段 |
| Rerank（重排） | 用更精的模型对初筛出的几十条结果重新打分排序，明显提升检索质量 |
| Function Calling / 工具调用 | 让模型按约定格式「点名」要调用的函数，是 AI 从「会说」到「会做」的关键 |
| Agent | 让模型自主规划步骤、循环调用工具直到完成任务的系统（不只是单轮问答） |
| 流式输出 | 生成一个字返回一个字（打字机效果），大幅降低体感等待时间 |
| TTFT / P99 | 首 Token 等待时间 / 99% 请求的最大延迟，AI 服务最看重的两个性能指标 |
| 微调（SFT/LoRA） | 用自己的数据继续训练模型改变其「习惯」；贵且慢，优先考虑 RAG——本计划暂不学 |

---

## 附录 B：本仓库六大阶段 vs 本计划的取舍

| 仓库阶段 | 本计划处理 | 原因 |
|---|---|---|
| 0 全局认知 | ✅ 纳入（Day 0 + 第 1 周前 2 篇） | 建立地图，成本低收益高 |
| 1 模型调用 | ✅ 完整纳入（第 1 周） | 一切的地基 |
| 2 RAG | ✅ 完整纳入（第 2-4 周，主线） | 你的目标项目 |
| 4 生产工程 | ✅ 简化纳入（第 5 周） | 项目要能上线；模型自部署部分（KV Cache/量化）暂跳过 |
| 3 Agent | ⚠️ 选修（第 6 周） | 进阶方向，主线达成后再学 |
| 5 Transformer 原理 | ❌ 暂不学 | 你选择了轻原理重项目；求职或想深入时再回来 |

> 计划是地图不是枷锁：某天状态差就少学点，但**别跳过动手环节**。
> 完成比完美重要，跑通比优雅重要。
