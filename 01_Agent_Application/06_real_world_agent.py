import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# =====================================================================
# 1. 基础配置
# =====================================================================
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# =====================================================================
# 2. 本地工具库 (Tools) - 实战新增：读取本地文件
# =====================================================================
def get_current_time():
    """获取当前系统时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_local_file(file_path: str):
    """实战工具：读取本地纯文本文件"""
    print(f"\n>>> [本地执行] 正在尝试读取文件: {file_path}")
    if not os.path.exists(file_path):
        return f"Error: 找不到文件 '{file_path}'，请确认路径是否正确。"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 工程容错 2：防止读取超大文件撑爆大模型上下文窗口
            max_chars = 3000
            if len(content) > max_chars:
                return f"文件过大，已截取前 {max_chars} 个字符：\n{content[:max_chars]}\n...[后续内容已截断]"
            
            return f"文件 '{file_path}' 的内容如下：\n{content}"
    except Exception as e:
        return f"读取文件时发生系统错误: {str(e)}"

def list_directory(dir_path: str = "."):
    """
    实战工具 2：列出指定目录下的所有文件和文件夹。默认查看当前目录。
    """
    try:
        if not os.path.exists(dir_path):
            return f"Error: 目录 '{dir_path}' 不存在。"
        
        files = os.listdir(dir_path)
        if not files:
            return f"目录 '{dir_path}' 是空的。"
            
        return f"目录 '{dir_path}' 下包含以下文件/文件夹：\n" + "\n".join(files)
    except Exception as e:
        return f"读取目录失败: {str(e)}"    

# 高内聚低耦合：工具路由映射字典
TOOL_MAP = {
    "get_current_time": get_current_time,
    "read_local_file": read_local_file,
    "list_directory": list_directory
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
            "name": "read_local_file",
            "description": "读取本地的纯文本文件（如 .txt, .md, .py, .tex, .log）。当用户要求你分析、总结或查阅某个本地文件时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "本地文件的相对路径或绝对路径，例如 'experiment_log.txt'"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的所有文件和文件夹。默认查看当前目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "要查看的目录路径，默认为当前目录 '.'"
                    }
                },
                "required": []
            }
        }
    }

]

# =====================================================================
# 3. Agent 核心引擎 (双向 HTTP 全量透视版)
# =====================================================================
def run_agent(user_prompt: str, max_steps: int = 5):
    messages = [{"role": "user", "content": user_prompt}]
    print(f"\n==================== [用户输入]: {user_prompt} ====================")

    step = 0
    while step < max_steps:
        step += 1
        print(f"\n#################### [ 第 {step} 轮 HTTP 交互 ] ####################")

        # 【全量透视 1】：拦截并打印发出去的请求包 (Request Payload)
        request_payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": tools
        }
        print("\n================ 📤 发出的请求包 (Request) ================")
        print(json.dumps(request_payload, indent=2, ensure_ascii=False))
        print("===========================================================\n")

        # 真正发起网络请求
        response = client.chat.completions.create(**request_payload)

        # 【全量透视 2】：拦截并打印收到的响应包 (Response Payload)
        print("\n================ 📥 收到的响应包 (Response) ===============")
        print(response.model_dump_json(indent=2))
        print("===========================================================\n")
        
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # 无论结果如何，必须将模型的回复追加进上下文缓存
        messages.append(response_message.model_dump(exclude_none=True))

        # 分支 A: 模型决策收敛，给出最终回答
        if finish_reason == "stop":
            final_answer = response_message.content
            print(f"\n✅ [Agent 最终回答]:\n{final_answer}\n" + "="*60)
            return final_answer

        # 分支 B: 模型要求调用工具，执行本地代码后继续循环
        elif finish_reason == "tool_calls":
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"🤖 [Action] 模型调度工具: `{func_name}` | 入参: {func_args}")
                
                # 动态路由执行
                if func_name in TOOL_MAP:
                    result = TOOL_MAP[func_name](**func_args)
                else:
                    result = f"Error: 找不到工具 {func_name}"
                
                # 将结果包装追加回历史记录中
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
    
    print("\n⚠️ 达到最大循环步数，引擎为防止死循环已强制中断。")
    return None

# =====================================================================
# 4. 测试运行
# =====================================================================
if __name__ == "__main__":
    # 真实场景测试
    prompt = "帮我用一句话总结我今天的实验遇到了什么核心问题？"
    run_agent(prompt)