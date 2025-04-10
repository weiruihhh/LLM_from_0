from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#导入环境初始化llm
import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
llm = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=qwen_api_key,	# app_key
    model_name="qwen-plus",	# 模型名称
)

#构建提示词
prompt_template = ChatPromptTemplate.from_template("""在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是新浪财经A股上市公司的公司简介。
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Extracted:"""
)


chain = prompt_template | llm

inputs = {
  "url": "https://finance.sina.com.cn/realstock/company/sz300059/nc.shtml"
}
response = chain.invoke({"requests_result":inputs})
# print(response.content)

#将获取到的json字符串保存到文件
import json
# 获取纯字符串
json_str = response.content.strip().replace("Extracted:", "").strip()

# 转换为字典
data = json.loads(json_str)
file_path = "./结构化网页爬虫/JSON/company_stock_profiles.json"

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