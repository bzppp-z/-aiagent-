"""Day 24 下半场: LLM 裁判 —— 生成指标（忠实度/正确性）+ 拒答率

链路位置: golden_set.json → [生产链路真答一遍] → 引用正则 + LLM 裁判 → 生成指标报告
用法: python eval/judge_eval.py

维度定义:
  correctness  —— 回答与标准答案的要点一致性（覆盖要点/无关键错误）
  faithfulness —— 回答是否忠实于【检索到的资料】（资料里有但标准答案里没有的正确细节不算编造）
  outside 类   —— 不进裁判，规则判定「是否正确拒答」（拒答标志词）

结构化输出: 裁判必须返回 JSON（chat() 的 response_format 参数，week1 Day 4 学的），
           散文没法算平均分——这是 LLM-as-judge 能落地的关键。
v2 修订: 裁判输入加入【检索到的资料】——v1 只对标准答案判忠实度，把"语料里有、
         标准答案里没有"的正确细节误判为编造（F001 的产品手册案例被冤枉）。
注意: 全量约 32 题 × 2 次 API 调用，几分钟跑完，成本几分钱量级（DeepSeek 便宜）。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
import json
import re
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from sentence_transformers import SentenceTransformer, CrossEncoder
from retrieval import HybridRetriever
from rag import build_messages
from chat import chat
from embed import MODEL_NAME
from rag import RERANKER, POOL
from run_eval import topk_with_scores

K = 5
GOLD = os.path.join(ROOT, "eval", "golden_set.json")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")

# 拒答标志词（outside 类判定用）
REFUSAL_MARKERS = ["没有相关内容", "资料中没有", "无法回答", "没有关于", "不在资料", "没有找到", "模型本身不知道"]

JUDGE_SYSTEM = (
    "你是严格的 RAG 问答评测裁判。根据【问题】【检索到的资料】【标准答案】评估【助教回答】。"
    "按以下定义打分（整数 1-5）：\n"
    "faithfulness：回答是否忠实于【检索到的资料】——回答中的具体断言、数字、名词应能在资料中找到依据或由其合理推断；"
    "资料与问题都撑不起的凭空细节（不存在的数字、人名、步骤）为低分；拒答类回答此项给 5。"
    "注意：资料里有但标准答案里没有的正确细节不算编造。\n"
    "correctness：回答与标准答案的要点一致程度——覆盖主要要点、无与标准答案矛盾的关键错误为高分；"
    "遗漏主要要点或给出矛盾说法为低分。\n"
    "verdict：两者均 ≥4 为 pass，任一 =3 为 warn，任一 ≤2 为 fail。\n"
    "reason：一句话中文说明扣分点或亮点。\n"
    '只输出 JSON 对象，形如 {"faithfulness": 4, "correctness": 5, "verdict": "pass", "reason": "..."}'
)


def judge_one(question, golden_answer, assistant_answer, context):
    """裁判一次：返回解析后的 dict（失败则给 parse_error 结构）。"""
    user = (f"【问题】{question}\n\n【检索到的资料】{context}"
            f"\n\n【标准答案】{golden_answer}\n\n【助教回答】{assistant_answer}")
    resp = chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},   # 强制 JSON（week1 结构化输出）
    )
    text = resp["text"].strip()
    try:
        return json.loads(text), resp["usage"]
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group()), resp["usage"]
            except json.JSONDecodeError:
                pass
    return {"faithfulness": 0, "correctness": 0, "verdict": "fail", "reason": "裁判输出无法解析"}, resp["usage"]


def is_refusal(answer):
    """outside 类判定：命中任一拒答标志词即视为正确拒答。"""
    return any(m in answer for m in REFUSAL_MARKERS)


def main():
    gold = json.load(open(GOLD, encoding="utf-8"))
    cases = gold["cases"]

    model = SentenceTransformer(MODEL_NAME)
    cross = CrossEncoder(RERANKER, max_length=512)
    hr = HybridRetriever(model)

    judged, refused = [], []
    for c in cases:
        print(f"  {c['id']} [{c['type']}] 处理中...", flush=True)
        top, _ = topk_with_scores(c["question"], hr, cross)
        hits = [(s, t) for s, t, _ in top]

        if c["type"] == "outside":
            # 生产链路照常回答（不 special-case，测真实行为），再规则判定拒答
            msgs = build_messages(c["question"], hits)
            result = chat(msgs, temperature=0.1)
            ok = is_refusal(result["text"])
            refused.append({
                "id": c["id"], "question": c["question"],
                "refusal_ok": ok,
                "answer_head": result["text"][:60].replace("\n", " "),
            })
            print(f"    拒答判定: {'✅ 正确拒答' if ok else '❌ 未正确拒答'}", flush=True)
            continue

        # fact / reasoning: 真答一遍 → 正则引用检查 → 裁判（带资料）
        msgs = build_messages(c["question"], hits)
        result = chat(msgs, temperature=0.1)
        answer = result["text"]
        has_citation = bool(re.search(r"\[\d+\]", answer))

        context = "\n\n".join(f"[{i}] ({s})\n{t}" for i, (s, t) in enumerate(hits, 1))
        verdict, judge_usage = judge_one(c["question"], c["answer"], answer, context)

        judged.append({
            "id": c["id"], "type": c["type"],
            "faithfulness": verdict.get("faithfulness"),
            "correctness": verdict.get("correctness"),
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "has_citation": has_citation,
            "gen_tokens": result["usage"],
        })
        print(f"    忠实度 {verdict.get('faithfulness')} / 正确性 {verdict.get('correctness')} "
              f"/ {verdict.get('verdict')} / 引用 {has_citation}", flush=True)

    # ---- 汇总 ----
    n = len(judged)
    mean_f = sum(d["faithfulness"] for d in judged) / n
    mean_c = sum(d["correctness"] for d in judged) / n
    verdicts = {}
    for d in judged:
        verdicts[d["verdict"]] = verdicts.get(d["verdict"], 0) + 1
    cite_rate = sum(d["has_citation"] for d in judged) / n
    refuse_rate = sum(r["refusal_ok"] for r in refused) / len(refused)

    print(f"\n===== 生成指标（LLM 裁判 v2 带资料，{n} 题 fact+reasoning） =====")
    print(f"忠实度均值: {mean_f:.2f} / 5")
    print(f"正确性均值: {mean_c:.2f} / 5")
    print(f"verdict 分布: {verdicts}")
    print(f"引用率: {cite_rate:.1%}")
    warn_fail = [d for d in judged if d["verdict"] in ("warn", "fail")]
    print("待复盘明细（warn/fail）:")
    for d in warn_fail:
        print(f"  {d['id']} [{d['verdict']}] 忠实{d['faithfulness']} 正确{d['correctness']} | {d['reason']}")

    print(f"\n===== 拒答率（outside {len(refused)} 题） =====")
    print(f"正确拒答率: {refuse_rate:.1%} ({sum(r['refusal_ok'] for r in refused)}/{len(refused)})")
    for r in refused:
        if not r["refusal_ok"]:
            print(f"  ❌ {r['id']} 回答开头: {r['answer_head']}")

    report = {
        "date": str(datetime.date.today()),
        "config": {"pipeline": "hybrid+rerank", "k": K, "judge_model": "deepseek-chat",
                   "judge_version": "v2-with-context", "cases": len(cases)},
        "generation": {"faithfulness_mean": round(mean_f, 3), "correctness_mean": round(mean_c, 3),
                       "verdicts": verdicts, "citation_rate": cite_rate, "detail": judged},
        "refusal": {"rate": refuse_rate, "detail": refused},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"judge_report_{report['date']}_v2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[报告] 已保存 {out}")


if __name__ == "__main__":
    main()
