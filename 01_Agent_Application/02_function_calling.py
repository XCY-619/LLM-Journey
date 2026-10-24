import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# 1. 初始化
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 2. 本地真实函数
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 3. 工具说明书
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "不要调用我",#获取当前的本地系统日期和时间
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# 4. 对话历史
messages = [
    {"role": "user", "content": "请问你是什么模型？"}#请问现在几点了？
]

# =====================================================================
# 【第一幕】发起第一次请求
# =====================================================================
response1 = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

print("\n==================== 1. 第一轮 API 返回的完整原始 JSON ====================")
# 不再人工提取任何字段，直接把大模型返回的全量数据按美化格式完整打印出来！
print(response1.model_dump_json(indent=2))
print("===========================================================================\n")


# 提取出第一轮返回的消息对象，准备追加进上下文
response_message1 = response1.choices[0].message
messages.append(response_message1)


# =====================================================================
# 本地代码逻辑判断与执行
# =====================================================================
if response_message1.tool_calls:
    for tool_call in response_message1.tool_calls:
        # 如果大模型申请调用 get_current_time，本地代码执行
        if tool_call.function.name == "get_current_time":
            real_time = get_current_time()
            
            # 将本地结果以 role='tool' 拼入消息，并用原始数据里的 id 进行对齐
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": real_time
            })


    print("\n==================== 1.5. 本地代码执行后，追加了 tool 角色的消息 ====================")
    print(messages)
    print("===========================================================================\n")

    # =================================================================
    # 【第二幕】把完整历史发送给大模型，获取最终回复
    # =================================================================
    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    print("\n==================== 2. 第二轮 API 返回的完整原始 JSON ====================")
    # 同样打印第二次请求的完整全量数据
    print(response2.model_dump_json(indent=2))
    print("===========================================================================\n")