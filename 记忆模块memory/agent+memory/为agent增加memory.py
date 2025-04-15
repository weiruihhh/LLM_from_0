from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory

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

#设置记忆模块
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#搜索工具
search = DuckDuckGoSearchResults(output_format="string")

#把duckduckgo的搜索结果封装成一个agent可以调用的工具
search_tool = Tool(
    name="duckduckgo-search",
    func=search.run,
    description="使用 duckduckgo 搜索来回答一般性问题"
)

tools = [search_tool]
# 创建 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,#添加记忆模块
    agent_kwargs={
        "prefix": """你是一个善良热心的助手.

当你遇到很有可能需要用到工具解决的问题时，你可以使用工具来帮助用户解决问题:

特别是当用户问到你和时间、天气、新闻、体育等明显用搜索引擎会得到更好答案的问题时,你就可以调用duckduckgo-search;但如果用户问的不相关,你就不要调用tools.你可以直接回答用户的问题.你会记住和用户对话的{chat_history}"""
    }
)

# 让 Agent 回答问题
response = agent.invoke({"input":"战国时期楚国的公子比,公子弃疾的故事是什么样的"})
print(response)

response2 = agent.invoke({"input":"我上一条问的是什么？"})
print(response2)