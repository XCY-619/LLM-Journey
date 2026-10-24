import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# 1. 环境初始化
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# =====================================================================
# 2. 本地工具库定义
# 定义两个工具，体验多步或组合调用的自动化过程
# =====================================================================
def get_current_time():
    """获取本地系统时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a: float, b: float):
    """计算两数之和"""
    return str(a + b)

# 核心机制 1：动态路由表（TOOL_MAP）
# 用字典将工具名称映射到真正的 Python 函数，避免写死 if/else
TOOL_MAP = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers
}

# 工具说明书列表
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的本地系统日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "计算两个数字相加，返回两数之和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# =====================================================================
# 3. 核心：ReAct 自动化循环引擎
# =====================================================================
def run_react_agent(user_prompt: str, max_steps: int = 5):
    """
    ReAct 自动循环引擎
    :param user_prompt: 用户输入的原始问题
    :param max_steps: 安全阈值，防止死循环无限消耗 Token
    """
    messages = [{"role": "user", "content": user_prompt}]
    step = 0

    print(f"\n==================== 收到用户任务: {user_prompt} ====================")

    # 核心机制 2：while 自动化循环
    while step < max_steps:
        step += 1
        print(f"\n-------------------- [ 循环第 {step} 轮 ] --------------------")

        # 1. 发起 API 请求
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        print("\n==================== 1. 模型接收到的完整原始 JSON ====================")
        print(messages)
        print("===========================================================================\n")

        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"当前轮次 finish_reason: '{finish_reason}'")

        # 2. 无论模型是说话还是调工具，都先追加进对话历史
        messages.append(response_message)

        print("\n==================== 2. 模型输出的消息内容 ====================")
        print(response_message.content)
        print("===========================================================================\n")

        # 3. 判定分支 A：模型输出终止，给出了最终答案
        if finish_reason == "stop":
            print("\n✅ Agent 决策收敛，完成任务！")
            print(f"最终回答: {response_message.content}")
            return response_message.content

        # 4. 判定分支 B：模型要求调用工具
        elif finish_reason == "tool_calls":
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                # 解析模型传过来的 JSON 参数字符串（反序列化为 Python 字典）
                func_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id

                print(f"🤖 [Thought -> Action] 模型决定调用工具: {func_name}")
                print(f"   解析到的入参: {func_args}")
                print(f"   分配的取餐号 ID: {tool_call_id}")

                # 动态匹配并执行本地函数
                if func_name in TOOL_MAP:
                    target_func = TOOL_MAP[func_name]
                    # **func_args 解包字典传入函数参数 (例如 a=10, b=20)
                    execution_result = target_func(**func_args)
                else:
                    execution_result = f"Error: 未定义的工具 {func_name}"

                print(f"👁️  [Observation] 本地执行结果: {execution_result}")

                # 将执行结果包装为 role='tool' 追加进对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(execution_result)
                })

            # 循环不中断，自动进入下一轮 while，带上新追加的 tool 结果再次发给大模型...

    print("⚠️ 已达到最大循环限制次数，强制安全退出。")

# =====================================================================
# 4. 测试运行
# =====================================================================
if __name__ == "__main__":
    # 测试场景：同时包含查时间与做计算的组合任务
    run_react_agent("帮我算一下 1234 加 5678 等于多少？再用这个值加上1038，最后告诉我值是多少。")