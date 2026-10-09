"""Day 2 地基模块：把「模型调用 + 输出处理」两步工程化。

对外只暴露一个 chat() 函数，返回 {"text": 回答, "usage": token 用量}。
后续所有代码（cli、RAG、服务化）都以它为地基。
"""
import os, time
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError

load_dotenv()  # 读取 .env，API Key 永远不写进代码

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",   # 换 GLM/Qwen 只改这一行
)


def chat(messages: list[dict], temperature: float = 0.1,
         max_tokens: int = 1024, retries: int = 2) -> dict:
    """返回 {"text": 回答, "usage": token 用量}；可恢复错误指数退避重试"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30,                      # 必须设超时，别裸等
)
            usage = resp.usage
            return {
                "text": resp.choices[0].message.content,
                "usage": {"in": usage.prompt_tokens, "out": usage.completion_tokens},
            }
        except (APITimeoutError, APIError) as e:
            last_err = e
            time.sleep(2 ** attempt)             # 指数退避：1s, 2s, 4s
    raise last_err
