from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
# 加载 llm 模型
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key
    model="qwen-plus",	# 模型名称
    max_completion_tokens=2048,
    temperature=0,
)

#messages包括system messages、humanmessages、ai messages 和 chat messages
# 分别表示场景消息、用户消息、模拟ai回复的消息和比较通用的chat消息,它可以人为指定role
messages = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob."
    ),
    HumanMessage(
        content="What is your name?"
    )
]

msg = [
    ChatMessage(role="system", content="你是一个幽默的 AI 助手"),
    ChatMessage(role="user", content="讲个笑话"),
]
response = llm.invoke(messages)
print(response.content)
response = llm.invoke(msg)
print(response.content)