"""Day 3 动手：结构化 Prompt + 「坏 Prompt vs 好 Prompt」对比。

坏 Prompt：只有一条 user 消息，没规则、没输出格式。
好 Prompt：system 定人设 + 规则 + 输出格式，user 只放本次数据。

跑完把两段输出贴进 README.md，写下你的差异观察。
"""
from chat1 import chat

# ① 角色 ② 任务 ④ 约束 ⑤ 输出格式 —— 都写进 system（不变的部分，可复用）
SYSTEM_TEMPLATE = """你是资深数据库工程师，专门帮用户诊断 SQL 执行报错。

任务：根据用户提供的 SQL 方言、报错信息和 SQL 语句，定位根因并给出修复方案。

规则：
1. 只依据用户提供的信息分析；缺少关键信息（如表结构、字段类型）时先追问，不得猜测；
2. 不得编造用户未提供的表名、字段名或函数；
3. 结论要明确；不确定时直接说"信息不足，需要…"，不要用"可能""也许"糊过去。

输出格式（严格按下面三段，不要输出多余文字）：
【根因】一句话说明报错原因
【修复】给出可执行的完整 SQL
【改动说明】1-2 句解释改了什么"""

# ③ 上下文数据 —— 变化的部分，运行时用变量替换注入
USER_TEMPLATE = """SQL 方言：{dialect}
报错信息：{error}
SQL 语句：
{sql}"""

# 本次要诊断的样例（换一套数据也能跑）
SAMPLE = {
    "dialect": "MySQL 8.0",
    "error": "Unknown column 'user_name' in 'field list'",
    "sql": "SELECT user_name, email FROM users WHERE id = 1;",
}

user_msg = USER_TEMPLATE.format(**SAMPLE)

# 坏 Prompt：不给 system，模型只能靠猜
bad = chat([{"role": "user", "content": user_msg}])

# 好 Prompt：system 定规则 + 输出格式
good = chat([
    {"role": "system", "content": SYSTEM_TEMPLATE},
    {"role": "user", "content": user_msg},
])

havefewshot=chat([
    {"role":"system","content":SYSTEM_TEMPLATE},
    # 示例 1：能直接修 —— 示范"修复"字段该怎么写
    {"role": "user", "content": "SQL 方言：PostgreSQL 14\n报错信息：syntax error at or near \"FROM\"\nSQL 语句：\nSELECT id name FROM users;"},
    {"role": "assistant", "content": "【根因】SELECT 列表里 id 和 name 之间缺少逗号。\n【修复】SELECT id, name FROM users;\n【改动说明】在 id 和 name 之间补上逗号。"},
    # 示例 2：信息不足 —— 示范"不猜测、先追问"（对应规则 1）
    {"role": "user", "content": "SQL 方言：MySQL 8.0\n报错信息：Table 'shop.orders' doesn't exist\nSQL 语句：\nSELECT * FROM orders;"},
    {"role": "assistant", "content": "【根因】当前连接的库 shop 中不存在 orders 表。\n【修复】无法给出可执行 SQL：无法确认表名是否写错，或该表是否位于其他库。\n【改动说明】请先执行 SHOW TABLES; 确认表名，或用 SHOW CREATE TABLE orders; 核对表名与所属库。"},
    {"role":"user","content":user_msg},
])

print("=" * 50)
print("【坏 Prompt】没有 system")
print("=" * 50)
print(bad["text"])
print(f"token: {bad['usage']}")

print()
print("=" * 50)
print("【好 Prompt】system 定规则 + 输出格式")
print("=" * 50)
print(good["text"])
print(f"token: {good['usage']}")

print()
print("=" * 50)
print("【好 Prompt】system 定规则+few-shot + 输出格式")
print("=" * 50)
print(havefewshot["text"])
print(f"token: {havefewshot['usage']}")
