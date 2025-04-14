# 🤖 LangChain + Agent 简明总结

## 📌 什么是 LangChain Agent？

在 LangChain 中，Agent 是一种智能体，它利用大型语言模型（LLM）作为推理引擎，根据用户的输入**自动选择和调用合适的工具（tools）**来完成任务。​通过 initialize_agent 和 AgentType，你可以快速构建一个具备多种功能的智能体。
简而言之：
> “你问的问题 -> 大语言模型思考后选择合适的工具 -> 获取结果 -> 返回答案”

---

## 🧠 Agent 的核心功能

- **动态调用工具**：根据需求自动决定调用哪个工具（如搜索引擎、计算器、本地函数等）。
- **多步推理**：支持 Chain-of-Thought，自动多轮调用多个工具组合完成任务。
- **可解释输出**：能清晰展示思考过程和每一步调用结果（如果设置 verbose=True）。

---

## Agent常用可调用工具分类
- **搜索引擎(Search)**
- **数据库分析(database)**
- **网页自动化浏览(web browser)**
  
    [langchain官方列举的工具](https://python.langchain.com/docs/integrations/tools/)
