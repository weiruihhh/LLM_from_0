from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key
    model="qwen-plus",	# 模型名称
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history_messages"),#message_placeholder是一个占位符，用于在运行时动态插入消息历史记录
    ("human", "{human_input}"),#这里的human_input和ability是等待用户在llm指定的输入
])

chain = prompt | llm

# 这里的session_id可以是任意字符串，作为对话的唯一标识,store是一个字典，用于存储不同会话的消息历史记录
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="human_input",
    history_messages_key="history_messages",
)

response = chain_with_memory.invoke(
    {"human_input": "你能帮我介绍一下鬼灭之刃吗？", "ability": "动漫专家"},
    config={"configurable": {"session_id": "123"}}#这个参数名称也是有讲究的，必须就是configurable
)
print(response.content)
response2 = chain_with_memory.invoke(
    {"human_input": "我上一个问题是什么？", "ability": "动漫专家"},
    config={"configurable": {"session_id": "123"}}#这个参数名称也是有讲究的，必须就是configurable
)
print(response2.content)