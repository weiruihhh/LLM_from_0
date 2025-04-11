import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

#配置llm
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
llm = ChatOpenAI(
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=qwen_api_key,	# app_key
    model_name="qwen-plus",	# 模型名称
    max_tokens=2048,
    temperature=0,
)
prompt = PromptTemplate.from_template(
    "以{topic}为主题，面向{recipient}写一份邮件 ."
)
#渲染 prompt
user_input = {
    "recipient": "公司HR",
    "topic": "大模型岗位求职"
}
final_prompt = prompt.format(**user_input)

# LLM 生成邮件正文
email_body = llm.invoke(final_prompt)

#配置gmail工具
from langchain_google_community import GmailToolkit
toolkit = GmailToolkit()
tools = toolkit.get_tools()
# print(tools)
GmailCreateDraft = toolkit.get_tools()[0]# 创建草稿但不发送邮件
GmailSendMessage = toolkit.get_tools()[1]# 直接发送邮件
GmailSearch = toolkit.get_tools()[2]# 搜索邮件
GmailGetMessage = toolkit.get_tools()[3]# 获取邮件
GmailGetThread = toolkit.get_tools()[4]# 获取邮件线程

"""在草稿箱中创建邮件"""
# GmailCreateDraft.run({
#         "to": ["1820046254@qq.com"],# 收件人
#         "subject": "Meeting Reminder",# 主题
#         # "cc":# 抄送
#         "message": email_body.content
# })

"""发送邮件"""
# GmailSendMessage.run({
#         "to": ["1820046254@qq.com"],# 收件人
#         "subject": "大模型求职申请",# 主题
#         # "cc":# 抄送
#         "message": email_body.content
# })

#搜索和获取邮件结果类似，只是一个是根据条件搜索，一个是根据id获取
"""搜索邮件"""
# emails = GmailSearch.run({
#     "query": "大模型求职申请"
# })
# print(emails)

#这两个有些鸡肋，因为message_id和thread_id是邮件的id，需要从搜索邮件结果中获取
"""获取邮件"""
# email_detail = GmailGetMessage.run({
#     "message_id": "19615f09e5bfe7ae"
# })
# print(email_detail)

"""获取邮件线程"""
# thread_detail = GmailGetThread.run({
#     "thread_id": "19615f09e5bfe7ae"
# })
# print(thread_detail)