from turtle import st
from langchain_core.output_parsers import BaseOutputParser
import csv
import io
from typing import List, Dict
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

#自定义CSV解析器
class CSVOutputParser(BaseOutputParser):
    """解析 LLM 返回的 CSV 文本为字典列表"""
    
    def parse(self, text: str) -> List[Dict[str, str]]:
        # 清理模型返回中的前后空行
        cleaned_text = text.strip()
        # 用 StringIO 模拟文件对象
        reader = csv.DictReader(io.StringIO(cleaned_text))
        return list(reader)

    @property
    def _type(self) -> str:
        return "csv"

# 构造 Prompt
prompt = ChatPromptTemplate.from_template(
    """请用 CSV 格式列出三种动物的名称、分类（哺乳类、爬行动物等）和栖息地，格式如下：
Name,Type,Habitat"""
)

# 执行 LLM
messages = prompt.format_messages()
response = llm.invoke(messages)

# 打印原始输出
print("模型输出：")
print(response.content)

# 使用我们自定义的 CSVOutputParser 解析
parser = CSVOutputParser()
structured_data = parser.parse(response.content)

print("\n结构化结果：",structured_data)
# for item in structured_data:
#     print(item)