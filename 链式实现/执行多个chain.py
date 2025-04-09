from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory



#导入环境
import os
from dotenv import load_dotenv
from networkx import overall_reciprocity
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
# 加载 llm 模型
llm = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=qwen_api_key,	# app_key
    model_name="qwen-plus",	# 模型名称
    max_tokens=2048,
    temperature=0,
)
#设置输出格式
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# location 链
location_prompt = ChatPromptTemplate.from_template(
        """你的目标是根据用户所提出的位置给出该位置下最经典的菜单.
        % 用户地理位置
        {user_location}

        你的回复:
        """
    )
location_chain = location_prompt | llm | output_parser 
# meal 链
meal_prompt = ChatPromptTemplate.from_template(
        """给你一份食物，请你给出居家实现它的简单食谱.
        % 食物
        {user_meal}

        你的回复:
        """
    )
#小链的格式是prompt | llm | output_parser可以忽略最后一个输出格式
meal_chain = meal_prompt | llm | output_parser

# 组合链，把前一个输出作为下一个输入
overall_chain = location_chain | meal_chain
review = overall_chain.invoke({"user_location":"南京"})
# print(review)