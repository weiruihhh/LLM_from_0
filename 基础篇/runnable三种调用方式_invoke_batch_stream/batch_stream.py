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
prompt = ChatPromptTemplate.from_template("请翻译成中文：{text}")
chain = prompt | llm
responses = chain.batch([
    {"text": "Hello"},
    {"text": "Goodbye"},
    {"text": "Thank you"},
])
for r in responses:
    print(r.content)
#stream就是流式逐字输出
for chunk in chain.stream({"text": "How are you?"}):
    print(chunk.content, end="", flush=True)