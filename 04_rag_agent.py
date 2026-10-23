import json
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# =====================================================================
# 1. 知识库准备
# =====================================================================
print("正在初始化本地向量数据库...")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="private_knowledge")

collection.add(
    documents=[
        "2026年实验室的最新规定：所有人离开实验室前必须关闭服务器机房的备用空调。",
        "我的硕士毕业论文核心方向是：基于二分图的聚类集成公平性研究。",
        "学校食堂二楼的糖醋排骨非常抢手，只在每周三中午 11:30 准时供应。"
    ],
    ids=["doc1", "doc2", "doc3"]
)
print("知识库加载完成！\n")

def search_knowledge_base(query_text: str):
    print(f"\n>>> [本地执行] 正在向量库匹配与 '{query_text}' 接近的片段...")
    results = collection.query(query_texts=[query_text], n_results=1)
    if results['documents'] and results['documents'][0]:
        return f"检索到的本地知识: {results['documents'][0][0]}"
    return "未找到相关内容。"

TOOL_MAP = {"search_knowledge_base": search_knowledge_base}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "本地知识库搜索工具。当遇到私密、本地规定、个人资料等问题时，提取核心关键词进行搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "完整问题进行搜索"}
                },
                "required": ["query_text"]
            }
        }
    }
]

# =====================================================================
# 2. ReAct 循环引擎 (请求包与响应包双向全透视版)
# =====================================================================
def run_react_agent(user_prompt: str, max_steps: int = 5):
    messages = [{"role": "user", "content": user_prompt}]
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n#################### [ 第 {step} 轮 HTTP 交互 ] ####################")

        # -----------------------------------------------------------
        # 【拦截 1】：构造并打印发出去的“请求包 (Request Payload)”
        # -----------------------------------------------------------
        request_payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": tools
        }
        print("\n================ 📤 发出的请求包 (Request) ================")
        print(json.dumps(request_payload, indent=2, ensure_ascii=False))
        print("===========================================================\n")

        # 真正发起网络请求，**request_payload 会将字典解包为关键字参数传入
        response = client.chat.completions.create(**request_payload)

        # -----------------------------------------------------------
        # 【拦截 2】：打印收到的“响应包 (Response Payload)”
        # -----------------------------------------------------------
        print("\n================ 📥 收到的响应包 (Response) ===============")
        print(response.model_dump_json(indent=2))
        print("===========================================================\n")

        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # 将模型返回的消息体追加到我们的本地上下文缓存中
        messages.append(response_message.model_dump(exclude_none=True))

        if finish_reason == "stop":
            print("\n✅ 任务完成，退出循环。")
            return response_message.content

        elif finish_reason == "tool_calls":
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id

                execution_result = TOOL_MAP[func_name](**func_args) if func_name in TOOL_MAP else "Error"
                
                # 将本地执行结果包装为 tool 消息，追加到本地上下文缓存中
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(execution_result)
                })

if __name__ == "__main__":
    run_react_agent("我周三中午想去学校食堂吃饭，有什么推荐的菜吗？必须是周三才有的。")#我周三中午想去学校食堂吃饭，有什么推荐的菜吗？必须是周三才有的。