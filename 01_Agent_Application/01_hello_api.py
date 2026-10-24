import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 自动读取 .env 文件中的配置
load_dotenv()

# 2. 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 3. 向大模型发起一次最简单的对话请求
response = client.chat.completions.create(
    model="deepseek-chat",  # 根据你使用的服务商更换模型名称
    messages=[
        {"role": "user", "content": "你好！如果你收到了这条消息，请用一句话告诉我你的名字。"}
    ]
)

# 4. 打印返回内容（核心正反馈）
print("==== 大模型响应成功 ====")
print(response.choices[0].message.content)
print(response)