"""Day 2 实验 a：temperature 对比。

同一个问题，temperature = 0 和 1.5 各跑 3 次，观察回答的稳定性差异。
（不改 main.py，保持入口干净；这个文件用完可以删。）
"""
from chat1 import chat

QUESTION = "用一句话给一家咖啡店起个名字。"
messages = [{"role": "user", "content": QUESTION}]

for temp in (0, 1.5):
    print(f"\n{'=' * 44}")
    print(f"temperature = {temp}")
    print("=" * 44)
    for i in range(3):
        r = chat(messages, temperature=temp)
        print(f"[{i + 1}] {r['text']}")
