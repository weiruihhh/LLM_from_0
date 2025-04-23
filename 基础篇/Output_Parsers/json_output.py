from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

# 定义字段结构,这个字段也能搭配其他非json格式的解析器输出，但最好还是json格式
schemas = [
    ResponseSchema(name="emotion", description="情绪，如 positive, neutral, or negative"),
    ResponseSchema(name="language", description="语言种类，如 English, Chinese")
]

# 用 response_schemas 创建 parser
output_parser = StructuredOutputParser.from_response_schemas(schemas)
# 获取格式说明字符串（用在 prompt 中）
format_instructions = output_parser.get_format_instructions()
# 创建 prompt 模板
prompt = ChatPromptTemplate.from_template(
    "请分析以下句子，并按指定格式返回结果：\n{format_instructions}\n句子：{sentence}"
)

# 构建 prompt 消息
messages = prompt.format_messages(
    sentence="I am really disappointed in your service.",
    format_instructions=format_instructions
)

# 调用模型
response = llm.invoke(messages)

# 解析模型输出
parsed_output = output_parser.parse(response.content)

print(parsed_output)