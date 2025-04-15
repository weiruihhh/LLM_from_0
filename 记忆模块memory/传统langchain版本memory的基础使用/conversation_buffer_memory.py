from langchain.memory import ConversationBufferMemory
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
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# chain = prompt_template | llm | memory #目前管道符还不支持memory
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
)

# 进行对话，连续调用，llm会记住之前的对话
response_1 = chain.predict(human_input="你好，明天南京天气如何？")
print(response_1)

response_2 = chain.predict(human_input="我应该去南京哪个地方玩?")
print(response_2)