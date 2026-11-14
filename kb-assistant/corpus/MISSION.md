# Mission: 大模型应用开发 —— 做出一个可上线的 RAG 知识库问答系统

## Why
学习者是有 Python3 + 后端基础的程序员，想转入 AI 应用开发方向。第一个里程碑不是「懂大模型」，而是**亲手做出一个可评估、可上线的知识库问答（RAG）项目**——能给人演示、能讲清每个环节的取舍。

## Success looks like
- 6 周内完成 4 个递进交付物：命令行 AI 助手 → 能跑通的 RAG Demo → 可评估的知识库系统（评测集 + 指标基线 + Bad Case 本）→ FastAPI 服务化上线（压测报告）
- 能白板讲清：一次模型请求的完整链路、RAG 五环节各自解决什么问题、为什么这样选型
- 能用数字说话：检索命中率、忠实度、P99/TTFT、Token 成本

## Constraints
- 全职学习，每天 4 小时以上；6 周
- 无 AI 基础：原理讲到直觉层面即可，不推导数学
- 模型接入走国内 API（DeepSeek 首选，GLM/Qwen 备选）；Embedding 用本地 BGE 模型
- Windows 开发环境，Python3 + FastAPI
- 学习资料以本目录 `llm-master` 仓库（卡码笔记）为主教材，课件需标注原文出处

## Out of scope
- Transformer 数学推导与手写实现（第 6 周后可选回补）
- 微调（SFT/RLHF/LoRA）——遇到 Prompt 调不好的任务再说
- 自部署开源模型（vLLM/KV Cache/量化）——当前走 API，用不到
- 求职面试表达训练——主线项目完成后按需开启
