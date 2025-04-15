<!-- ConversationBufferMemory:存储完整的对话历史，不做任何压缩。最常见。
这个调用方式是最经典常用的方式，不过它一定需要LLMChain，而LLMChain是比较老的版本了

ConversationBufferWindowMemory:滑动窗口记忆；能保存k轮对话，再多就不记得了。
它具体调用上和ConversationBufferMemory一致，都需要LLMChain

ConversationSummaryBufferMemory:自动对内容进行总结，并不是存储完整的对话历史，这样做有助于节省token,适合没有太高历史对话精度的场景。
值得注意的是，这个方法的llm不能是 ChatOpenAI，它目前只支持openai的几个模型

ConversationTokenBufferMemory:与之前的类似，只是存储的方式从轮数变成了token数量，滑动token记忆

ConversationSummaryMemory:自动总结前文，更加常用

ChatMessageHistory:单纯用于记录对话历史（消息列表），一般作为 Memory 的一部分使用。


除了直接调用这些模块之外，还可以把记忆封装到agent里面，或者使用链+memory的方式。 -->

# 🧠 LangChain Memory 模块使用指南
LangChain 提供了多种 Memory（记忆）模块，用于记录和管理多轮对话历史，从而让 LLM 拥有“上下文记忆能力”。本文档将介绍常用的几种 Memory 类型、使用方式及适用场景并给出对应的示例。

## 📌 快速对比

| Memory 类型                      | 核心特性                      | 是否自动总结 | 控制方式       | 适用场景                    |
|----------------------------------|-------------------------------|--------------|----------------|-----------------------------|
| `ConversationBufferMemory`       | 存储完整对话，不做压缩         | ❌           | 全部存储       | 最常见，用于调试和短对话     |
| `ConversationBufferWindowMemory` | 仅保留最近 k 轮对话            | ❌           | 轮数窗口       | 多轮但只关注最近上下文       |
| `ConversationTokenBufferMemory`  | 仅保留最近 N 个 token          | ❌           | token 窗口     | token 敏感的对话            |
| `ConversationSummaryBufferMemory`| 总结历史 + 保留最近对话        | ✅           | 总结 + 滑动窗  | 节省 token，有一定上下文     |
| `ConversationSummaryMemory`      | 总结历史对话，无完整记录        | ✅           | 总结替代历史   | 极限节省 token，无需精度     |
| `ChatMessageHistory`             | 消息列表存储（用于自定义场景）  | ❌           | 自定义使用     | 自定义存储或结合其他模块使用 |

## 🧠 memory的使用方式

✅ 1. 搭配 LLMChain（传统方式）
```python
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_messages([...]),
    memory=ConversationBufferMemory(return_messages=True)
)
```
✅ 2. 搭配 Runnable（推荐新方式）
```python
chain = prompt | llm
memory = ConversationBufferMemory(return_messages=True, memory_key="history")
```

✅ 3. 搭配 agent使用（高级方式）
```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,#添加记忆模块
    agent_kwargs={
        "prefix": """你是一个善良热心的助手.
你会记住和用户对话的{chat_history}"""
    }
)
```

