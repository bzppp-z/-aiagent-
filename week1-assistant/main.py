"""Day 2 入口：调用 chat() 发起一次对话，打印回答与 token 用量。

在 VS Code 里打开本文件，点右上角 ▶ 运行即可。
"""
from chat1 import chat

result = chat([{"role": "user", "content": "你好，用一句话介绍你自己"}])

print("回答：", result["text"])
print("Token 用量：", result["usage"])
