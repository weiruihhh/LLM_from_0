from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()

# 添加 AI 和 用户的消息
history.add_user_message("你是谁？")
history.add_ai_message("我是一个 AI 聊天助手。")

# 查看历史
for msg in history.messages:
    print(msg.type, msg.content)
