# LangChain 中的链（Chain）机制总结
[链机制参考](https://python.langchain.com.cn/docs/expression_language/)
## 📌 链式执行的核心理念

链（Chain）是 LangChain 的核心抽象之一，其基本思想是**将多个模块组合成一个序列化的流程**，使得**前一个模块的输出可以直接作为下一个模块的输入**，从而实现类似流水线（pipeline）的处理方式。

这一设计使得 LangChain 在处理复杂的多步任务时尤为高效，尤其适用于需要中间变量、逐步推理、上下文累积等场景。


## 🔗 应用场景

链式结构的主要适用场景包括但不限于：

- 需要保留中间结果作为后续输入的任务
- 多步问答与推理
- 多模型协作（如不同 LLMs 处理不同子任务）
- 带记忆（Memory）的对话系统(一般很少用)


## 🧱 链的构建方式

LangChain 提供了多种方式来构建链：

### 1. 传统方式（函数式或类式定义）

最基础的方式是通过编程逻辑手动串联各个模块，如 PromptTemplate、LLM、Parser 等：

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate.from_template("Translate the following text to French: {text}")
llm = OpenAI()
chain = LLMChain(prompt=prompt, llm=llm)

result = chain.run("Hello, how are you?")
```
### 2. LCEL：LangChain Expression Language（强烈推荐）
LCEL 是 LangChain 最新推出的声明式语法，允许你通过使用 | 管道符号，直观地将多个组件连接成链，语义清晰、可组合性强。

例如：

```python
overall_chain = chain1 | chain2 | chain3
```
每一个小链也可以拆解成标准结构：

```python
single_chain = prompt | llm | output_parser | memory
#一般还是不用memory
single_chain = prompt | llm | output_parser
```
这种方式具有函数式编程的优雅风格，并且与可视化组件（如 LangSmith）完美兼容。

## 🔄 链的模块组件
一个链通常由以下标准组件组成：
| 组件             | 说明                                                        |
|------------------|-------------------------------------------------------------|
| `PromptTemplate` | 提示词模板，负责构建传递给 LLM 的输入                      |
| `LLM`            | 大语言模型模块，如 OpenAI、ChatGLM、Claude 等              |
| `OutputParser`   | 输出解析器，将 LLM 的返回结果转换为结构化数据或变量        |
| `Memory`         | 记忆模块，用于记录历史对话信息并注入上下文                |
