# eval/ — 评测闭环（v0.4）

## 文件

- `golden_set.json` — 评测集（Golden Set）：35 条 = 事实 19 + 推理 8 + 语料外 8

## 字段说明

| 字段 | 含义 |
|---|---|
| `type` | fact（语料内有标准答案）/ reasoning（需跨片段综合）/ outside（语料外，考拒答） |
| `question` / `answer` | 评测问题与标准答案（outside 类固定为「应拒答」） |
| `source` | **期望命中的文档**——Day 24 算 Recall@5 的判定依据（按文档名匹配检索 Top-5）；语料外题为 `[]` |
| `answer_verified` | **false = AI 起草、学员未校对——正式评测前必须逐条把关** |
| `origin` | 条目来源（bad_cases 编号 / 实测 / draft），便于追溯 |

## Day 22-23 校对作业（人把关流程）

1. 打开 golden_set.json，过滤 `answer_verified: false` 的 5 条（F009 / F010 / F020 / R005 / X008）；
2. 对照 `corpus_sources.json` 里的原文核实答案是否忠实、出处是否正确；
3. 核对无误后把该条改成 `true`——全部为 true 时评测集才算生效；
4. 有异议的答案直接改，改完在 `origin` 后追加 `（学员校对 2026-11-XX）`。

## 后续（Day 24-25 实现）

- `run_eval.py`：逐条跑检索 → 判定 source 是否进 Top-5 → 输出 Recall@5 / 按类型分组的命中率
- LLM 裁判：生成指标（忠实度/相关性），结构化输出保证 JSON 可解析
