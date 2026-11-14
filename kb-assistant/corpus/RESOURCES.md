# 大模型应用开发（RAG 主线）Resources

## Knowledge（主教材：本地仓库，路径相对于 `ai学习/`）

- [llm-master](../../llm-master/docs/README.md) — 卡码笔记·大模型全栈学习路线，150+ 篇中文教程。**主教材**，课件知识全部从此提炼，不从模型参数记忆编造。
  - 第 1 周用：`docs/roadmap/application.md` 指向的 10 篇应用开发文章
  - 第 2-4 周用：`docs/topics/rag.md` 主干 9 篇 + `docs/llm/app/rag_evaluation.md`
  - 第 5 周用：`docs/topics/deployment.md`（选读 3 篇）
  - 第 6 周用：`docs/roadmap/agent.md` 指向的 7 篇
- [完整学习路线](../../llm-master/docs/roadmap/README.md) — 六阶段总目录，课程结构与它对齐
- [全部教程索引](../../llm-master/docs/README.md) — 按问题查文章用

## Knowledge（官方文档，代码写法以此为准）

- [DeepSeek API 文档](https://api-docs.deepseek.com/zh-cn/) — 首选模型 API；OpenAI SDK 兼容，注意其不提供 Embedding 接口
- [智谱 GLM 开放平台](https://docs.bigmodel.cn/) — 备选 API；glm-4-flash 免费档；提供 Embedding-3 与 Rerank
- [阿里云百炼（Qwen）](https://help.aliyun.com/zh/model-studio/) — 备选 API；text-embedding-v3/v4、免费额度
- [OpenAI Python SDK](https://github.com/openai/openai-python) — 所有国产 API 的调用方式（改 base_url）
- [FastAPI 文档（中文）](https://fastapi.tiangolo.com/zh/) — 第 5 周服务化
- [ChromaDB 文档](https://docs.trychroma.com/) — 第 2 周向量库
- [BGE 模型（BAAI）](https://huggingface.co/BAAI/bge-small-zh-v1.5) — 本地 Embedding 模型，CPU 可跑；国内可从 modelscope 下载
- [pytest 文档](https://docs.pytest.org/en/stable/) — 第 1 周迷你项目测试

## Wisdom（Communities）

- 暂未选定。候选：卡码笔记公众号社区（主教材官方渠道）；V2EX / 即刻 AI 应用开发圈子；GitHub llm-master 仓库 Issues。首次遇到「教程没覆盖的问题」时再向用户推荐加入。
