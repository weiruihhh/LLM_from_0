from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')

llm = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=qwen_api_key,	# app_key
    model_name="qwen-plus",	# 模型名称
)

prompt_template = ChatPromptTemplate.from_template("""在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是NBA球员的信息。
请抽取参数请求的信息,完全按照网页上的准确信息来,不要更改网页上原有信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "player_name":"a",
  "height":"b",
  "weight":"c",
    "team":"d",
    "position":"e",
    "birth":"f",
    "birth_place":"g",
    "draft":"h"
}}
Extracted:"""
)

chain = prompt_template | llm

inputs = {
  "url": "https://nba.hupu.com/players/traeyoung-150951.html"
}

response = chain.invoke({"requests_result":inputs})
# print(response.content)


#将获取到的json字符串保存到文件
import json
# 获取纯字符串
json_str = response.content.strip().replace("Extracted:", "").strip()

# 转换为字典
data = json.loads(json_str)
file_path = "./结构化网页爬虫/JSON/NBA_player_profiles.json"

# 如果文件不存在或为空，创建一个空数组
if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
# 读取原有数据，追加新的内容
with open(file_path, "r+", encoding="utf-8") as f:
  content = json.load(f)
  content.append(data)
  f.seek(0)
  json.dump(content, f, ensure_ascii=False, indent=2)