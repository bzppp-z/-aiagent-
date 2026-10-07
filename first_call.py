"""Day 0 验收脚本：证明「环境 + API Key + 网络」全部就绪"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "用一句话证明你是一个大模型"}],
)

print("回答：", resp.choices[0].message.content)
print("Token 用量：输入", resp.usage.prompt_tokens,
      "/ 输出", resp.usage.completion_tokens)
