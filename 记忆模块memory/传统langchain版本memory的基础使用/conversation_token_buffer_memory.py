from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

#导入环境
import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
# 加载 llm 模型
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key
    model="qwen-plus",	# 模型名称
    temperature=0,
)

prompt_template = ChatPromptTemplate.from_template(
    """你是一个和人类对话的机器人助手.

    {chat_history}

    Human: {human_input}
    Chatbot:"""
    )
memory = ConversationTokenBufferMemory(
    llm=llm,  # 💡 你必须传入一个支持 token 计数的 LLM（如 ChatOpenAI）
    max_token_limit=1000,  # 💡 上下文最大 token 限制
    memory_key="chat_history",  # PromptTemplate 中占位符名字
    return_messages=True  # 若为 True，返回 Message 类型，适配 ChatPromptTemplate
)
# chain = prompt_template | llm | memory #目前管道符还不支持memory
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
)

# 进行对话，连续调用，llm会记住之前的对话,但只会保留设置的k轮
response_1 = chain.invoke({"human_input":"请你介绍一下中央财经大学"})
print(response_1.get("text"))

response_2 = chain.invoke({"human_input":"中央财经大学的校训是什么"})
print(response_2.get("text"))

response_3 = chain.invoke({"human_input":"我第一个问题问了什么"})
print(response_3.get("text"))