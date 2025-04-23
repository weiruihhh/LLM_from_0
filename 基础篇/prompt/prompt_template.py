from urllib import response
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
# 加载 llm 模型
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key # type: ignore
    model="qwen-plus",	# 模型名称
    max_completion_tokens=2048,
    temperature=0,
)
#ChatPromptTemplate主要使用包括from_template和from_message两种
#其中from_message功能更强，可以包含多轮对话,可指定多个角色；
#from_template一般就只是一轮简单问答，默认角色为human message
prompt = ChatPromptTemplate.from_template("请翻译以下句子：{sentence}")
message = prompt.format_messages(sentence="I love you")
response = llm.invoke(message)
print(response.content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译机器人。"),
    ("human", "请将以下句子翻译为英文：{sentence}")
])
messages = prompt.format_messages(sentence="我今天很开心")
response = llm.invoke(messages)
print(response.content)

#利用模版类
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一名擅长中英翻译的助手。"),
    HumanMessagePromptTemplate.from_template("请将以下句子翻译成英文：{sentence}")
])
messages = chat_prompt.format_messages(sentence="我今天心情很好。")
response = llm.invoke(messages) 
print(response.content)

#或者直接用message类
from langchain.schema import SystemMessage, HumanMessage, AIMessage
messages = [
    SystemMessage(content="你是一名擅长中英翻译的助手。"),
    HumanMessage(content="请将以下句子翻译成英文：我今天心情很好。")
]
response = llm.invoke(messages)
print(response.content)