# week1-assistant

第 1 周迷你项目：命令行 AI 助手（对应学习计划 Day 1–7）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `chat1.py` | Day 2 地基模块：封装模型调用（超时 / 重试 / token 统计），对外暴露 `chat()` |
| `main.py` | 最小入口：调用一次 `chat()` 并打印回答与 token |
| `exp_temperature.py` | Day 2 实验：对比 temperature = 0 与 1.5 的输出稳定性 |
| `prompt_lab.py` | Day 3 实验：结构化 Prompt 与「坏 Prompt vs 好 Prompt」对比 |
| `解析chat1.py代码语法.html` | Day 2 代码讲解 |
| `prompt_lab教学.html` | Day 3 代码讲解 |

---

## Day 3 · Prompt 工程化对比实验

**目的**：验证「把规则放进 system，输出是否更稳定、可控」。

**测试输入**（`USER_TEMPLATE` 填入 `SAMPLE`）：

- SQL 方言：MySQL 8.0
- 报错信息：`Unknown column 'user_name' in 'field list'`
- SQL 语句：`SELECT user_name, email FROM users WHERE id = 1;`

**控制变量**：两次调用的 user 消息完全相同，唯一差别是「有没有 system」。

### 坏 Prompt 输出（只有 user，无 system）

````text
# 错误分析

## 报错原因

`Unknown column 'user_name' in 'field list'` 表示 **`users` 表中不存在名为 `user_name` 的列**。

MySQL 在解析 `SELECT` 语句时，会先校验字段列表（field list）中的每一列是否存在于目标表中。只要有一列不存在，就会立即抛出该错误，**不会执行查询**。

## 排查步骤

### 1. 查看表结构，确认实际列名

```sql
DESC users;
-- 或
SHOW COLUMNS FROM users;
-- 或
SHOW CREATE TABLE users;
```

常见情况是列名拼写不同，例如：

| 你写的 | 实际可能是 |
|--------|-----------|
| `user_name` | `username` |
| `user_name` | `name` |
| `user_name` | `userName`（大小写敏感场景） |
| `user_name` | `nick_name` |

### 2. 确认当前数据库

```sql
SELECT DATABASE();
```

如果连错了库，可能查的是另一个同名但结构不同的 `users` 表。

### 3. 确认表名大小写

Linux 下 MySQL 默认 `lower_case_table_names=0`，表名区分大小写。确认 `users` 表确实存在：

```sql
SHOW TABLES LIKE 'users';
```

## 修复方案

假设实际列名是 `username`：

```sql
SELECT username, email FROM users WHERE id = 1;
```

如果确实需要 `user_name` 这个别名，可以用 `AS`：

```sql
SELECT username AS user_name, email FROM users WHERE id = 1;
```

## 常见坑点提醒

1. **`user_name` vs `username`**：这是最常见的拼写差异，很多 ORM 框架（如 MyBatis-Plus、JPA）默认下划线转驼峰，容易让人误以为数据库列名带下划线。
2. **保留字冲突**：如果列名是 `user`、`order`、`key` 等 MySQL 保留字，必须用反引号包裹，例如 `` `user` ``。但 `user_name` 不是保留字，所以本例不是这个原因。
3. **视图/别名表**：如果 `users` 是视图，字段可能来自 `SELECT ... AS xxx`，需查看视图定义。
4. **多表 JOIN 场景**：如果 SQL 里有 JOIN，字段可能属于另一张表，需要加表前缀，如 `u.user_name`。

## 快速定位命令

```sql
-- 一条命令看清所有列名
SELECT COLUMN_NAME, DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = DATABASE() 
  AND TABLE_NAME = 'users';
```

执行后对照结果，把 `user_name` 改成真实列名即可。
````

token: `{'in': 46, 'out': 623}`

### 好 Prompt 输出（system + 五段式）

```text
【根因】users 表中不存在名为 user_name 的列，MySQL 在解析字段列表时找不到该列。

【修复】无法给出可执行 SQL：缺少 users 表的实际结构，无法确定正确的列名。

【改动说明】需要你提供 users 表的建表语句（SHOW CREATE TABLE users;）或字段列表，确认用户名字段的真实名称（如 username、name、user_nick 等）后，才能改写 SQL。
```

token: `{'in': 205, 'out': 97}`

### 差异观察

| 维度 | 坏 Prompt（无 system） | 好 Prompt（system + 五段式） |
|---|---|---|
| 格式 | ❌ 无结构，标题/表格/代码块全凭模型心情 | ✅ 稳定按【根因】【修复】【改动说明】三段 |
| 是否编造 | ❌ **擅自假设**列名是 `username`，直接给出修复 SQL | ✅ 明确说「缺少表结构，无法给出可执行 SQL」，不猜 |
| 追问行为 | ❌ 不追问，自己往下走 | ✅ 主动要求补充 `SHOW CREATE TABLE users;` |
| 输出长度 | 623 tokens（一大篇） | 97 tokens（简洁） |
| 输入长度 | 46 tokens（无 system） | 205 tokens（含 system 规则） |
| 总 token | 669 | 302 |

### 结论

1. **不加 system，模型会自由发挥**：格式不可控、内容啰嗦，还会违反我们的业务规则（本例中它**擅自假设列名并给出修复 SQL**）。
2. **加上五段式 system 后**，模型稳定按三段输出，并且**遵守了「信息不足时先追问、不得猜测」的规则**——这正是我们想要的「可控性」。
3. **token 的账要算两头**：好 Prompt 的输入更大（多了 system 规则，+159 tokens），但因为输出被约束得简洁，**总 token 反而更少（669 → 302）**。规则不是白写的，它省在输出上。
4. 这印证了今天的核心：**规则归 system，数据归 user，模板与变量分离。**

### Few-shot 实验

**示例设计**（2 对，故意示范两种行为）：

| 示例 | 演示 | 对应 system 规则 |
|---|---|---|
| PostgreSQL `SELECT id name FROM users` | 错误明确 → 直接给出修复 SQL | 「给出可执行 SQL」 |
| MySQL `Table 'shop.orders' doesn't exist` | 信息不足 → 不猜，要求补充 | 规则 1「缺关键信息先追问」 |

代码（`prompt_lab.py`）：

```python
havefewshot = chat([
    {"role": "system", "content": SYSTEM_TEMPLATE},
    # 示例 1：能直接修 —— 示范"修复"字段怎么写
    {"role": "user", "content": "SQL 方言：PostgreSQL 14\n报错信息：syntax error at or near \"FROM\"\nSQL 语句：\nSELECT id name FROM users;"},
    {"role": "assistant", "content": "【根因】SELECT 列表里 id 和 name 之间缺少逗号。\n【修复】SELECT id, name FROM users;\n【改动说明】在 id 和 name 之间补上逗号。"},
    # 示例 2：信息不足 —— 示范"不猜测、先追问"
    {"role": "user", "content": "SQL 方言：MySQL 8.0\n报错信息：Table 'shop.orders' doesn't exist\nSQL 语句：\nSELECT * FROM orders;"},
    {"role": "assistant", "content": "【根因】当前连接的库 shop 中不存在 orders 表。\n【修复】无法给出可执行 SQL：无法确认表名是否写错，或该表是否位于其他库。\n【改动说明】请先执行 SHOW TABLES; 确认表名，或用 SHOW CREATE TABLE orders; 核对表名与所属库。"},
    {"role": "user", "content": user_msg},     # 真正要问的放最后
])
```

**输出**：

```text
【根因】users 表中不存在 user_name 列。
【修复】无法给出可执行 SQL：无法确认该列的真实名称。
【改动说明】请先执行 SHOW COLUMNS FROM users; 确认列名后再替换 user_name。
```

token: `{'in': 394, 'out': 53}`

**三版对比**：

| 版本 | system | Few-shot | in | out | 总 token |
|---|---|---|---|---|---|
| 坏 Prompt | ❌ | ❌ | 46 | 530 | 576 |
| 好 Prompt | ✅ | ❌ | 205 | 95 | 300 |
| 好 Prompt + Few-shot | ✅ | ✅ | 394 | 53 | 447 |

**结论**：

1. **Few-shot 的实际效果是「输出更简洁、措辞更贴示例」**：输出从 95 → 53 tokens。措辞明显靠拢示例 2——【修复】照用了"无法给出可执行 SQL：…"，【改动说明】写成"请先执行 SHOW COLUMNS FROM users; 确认…"，对应示例 2 的"请先执行 SHOW TABLES; 确认…"。
2. **代价是输入变长**：每次调用都要重发 2 对示例，输入 +189 tokens。本例净 token **不降反升**（300 → 447）。
3. **所以 Few-shot 值不值，要拿数据说话**：省的是输出、费的是输入；收益（一致性 / 简洁度）是否值这 +147 tokens，得看评测集，不能凭感觉。
4. **一个诚实的观察**：三版里，"守规则（不猜测、先追问）"其实**是 system 的功劳**——好 Prompt 版（无 Few-shot）已经守住了；Few-shot 版主要带来的是**措辞风格与简洁度**的一致。这说明在这个"规则已说清"的任务上，Few-shot 的增量有限，与预期一致。
