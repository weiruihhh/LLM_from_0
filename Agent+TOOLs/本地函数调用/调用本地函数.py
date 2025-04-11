from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key
    model="qwen-plus",	# 模型名称
)

@tool
def add_numbers(input: str) -> str:
    """Add two integers. Input should be a string like '3 and 5'."""
    try:
        parts = input.strip().split()
        nums = [int(s) for s in parts if s.isdigit()]
        if len(nums) != 2:
            return "Please provide exactly two numbers."
        return str(nums[0] + nums[1])
    except Exception as e:
        return f"Error: {str(e)}"
@tool
def subtract_numbers(input: str) -> str:
    """Subtract two integers. Input should be a string like '10 and 3'."""
    try:
        parts = input.strip().split()
        nums = [int(s) for s in parts if s.isdigit()]
        if len(nums) != 2:
            return "Please provide exactly two numbers."
        return str(nums[0] - nums[1])
    except Exception as e:
        return f"Error: {str(e)}"
    
tools = [add_numbers,subtract_numbers]
# 创建 agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """你是一个善良热心的助手.

当你遇到接近的问题时，你可以使用工具来帮助用户解决问题:

特别是当用户问到你类似"9和10相加等于几","请你计算1加2"之类的问题时,你就可以调用add_number;同理如果问到你相减的问题比如"8减2等于多少"，你就调用subtract_numbers;但如果用户问的不相关,你就不要调用tools.你可以直接回答用户的问题.还需要注意的如果要调用工具,那么在调用工具前,你要先提取出数字为字符串格式比'1 2'"""
    }
)

# agent.invoke({"input":"请帮我把 7 和 15 相加"})
# agent.invoke({"input":"请帮我把 98 和 15 相减"})
agent.invoke({"input":"请帮我把 98 和 15 相乘"})

