import json
import os
from datetime import datetime
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# =====================================================================
# 1. 基础配置与知识库初始化
# =====================================================================
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 初始化轻量级向量库 (可根据实际项目替换为连接真实库)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="private_knowledge")
collection.add(
    documents=[
        "2026年实验室的最新规定：所有人离开实验室前必须关闭机房备用空调。",
        "我的硕士毕业论文核心方向是：基于二分图的聚类集成公平性研究。"
    ],
    ids=["doc1", "doc2"]
)

# =====================================================================
# 2. 本地工具库 (Tools) 
# =====================================================================
def get_current_time():
    """获取当前系统时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a: float, b: float):
    """计算两数相加"""
    return str(a + b)

def search_knowledge_base(query_text: str):
    """RAG 本地知识检索"""
    results = collection.query(query_texts=[query_text], n_results=1)
    if results['documents'] and results['documents'][0]:
        return results['documents'][0][0]
    return "未找到相关内容"

# 工具映射路由（注册表）
TOOL_MAP = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
    "search_knowledge_base": search_knowledge_base
}

# 工具说明书 Schema
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
            "description": "计算两个数字相加",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "加数 A"},
                    "b": {"type": "number", "description": "加数 B"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "本地知识库搜索工具。用于查询实验室规定、论文等私有信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query_text"]
            }
        }
    }
]

# =====================================================================
# 3. Agent 核心自动化引擎 (标准生产级模板)
# =====================================================================
def run_agent(user_prompt: str, max_steps: int = 5):
    """
    轻量级无依赖 ReAct Agent 引擎
    """
    messages = [{"role": "user", "content": user_prompt}]
    print(f"\n👤 [User]: {user_prompt}\n" + "-"*50)

    for step in range(1, max_steps + 1):
        # 1. 提交完整历史给大模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )
        
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # 无论结果如何，必须将模型的回复追加进上下文缓存
        messages.append(response_message.model_dump(exclude_none=True))

        # -------------------------------------------------------------
        # 分支 A: 模型已搜集完毕信息，给出最终自然语言回答
        # -------------------------------------------------------------
        if finish_reason == "stop":
            final_answer = response_message.content
            print(f"\n🤖 [Agent 最终回答]:\n{final_answer}\n" + "="*50)
            return final_answer

        # -------------------------------------------------------------
        # 分支 B: 模型要求调用工具，执行后进入下一次循环
        # -------------------------------------------------------------
        elif finish_reason == "tool_calls":
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"🧠 [Agent 思考] 决定调用工具: `{func_name}` | 入参: {func_args}")
                
                # 动态路由并执行
                if func_name in TOOL_MAP:
                    result = TOOL_MAP[func_name](**func_args)
                else:
                    result = f"Error: 找不到工具 {func_name}"
                
                print(f"🛠️  [本地执行] 工具返回结果: {result}")
                
                # 将结果包装追加回历史记录中
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
    
    print("\n⚠️ 达到最大循环步数，引擎为防止死循环已强制中断。")
    return None

# =====================================================================
# 4. 综合测试
# =====================================================================
if __name__ == "__main__":
    # 我们用一个极限测试：同时包含无需工具(日常对话) + 查时间 + 算数 + 检索本地私有库 的综合任务
    prompt = "你好呀！请问现在几点了？帮我算一下 2.5 加 3.8 是多少？另外，我忘了硕士毕业论文的核心方向是什么了，能帮我查一下吗？"
    run_agent(prompt)