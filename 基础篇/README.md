<!-- 本章节主要参考青蛙书第一章
基础篇主要包括：prompt、output_parser输出格式、以及链chain的几种调用方法 -->

#   LangChain 的 LLM 基础篇
基础篇主要参考《Learning LangChain -- Mayo Oshin & Nuno Campos》青蛙书第一章

代码主要涉及到聊天对话角色(Humanmessage、Systemmessage 等)提示词模板 prompt template、输出格式 output parser、链chain的几种可运行接口

## 聊天消息类型
 **聊天消息类型 (Chat Message Types):** 与 `LLM` 交互时使用的关键元素：

*    `SystemMessage`: 设定 AI 角色的高级指令或背景信息，通常放在对话开头。
*   `HumanMessage`: 代表最终用户的输入或问题。
*    `AIMessage`: 代表 AI 模型的回复或输出。
*    `ChatMessage`: 是更通用的指定方式，可以人为指定对话的 role,包括 `system`、`human`、`ai`、`function`等


## 提示词模板（PromptTemplate）

在 LangChain 中，包括 PromptTemplate 和 ChatPromptTemplate 两种用于构造 输入提示（prompt） 的工具，但它们各自服务的对象和格式是不同的。

**本项目中所使用的均为 ChatPromptTemplate**
| 特性               | PromptTemplate                    | ChatPromptTemplate                                  |
|--------------------|-----------------------------------|-----------------------------------------------------|
| 面向模型类型       | 🔤 文本模型（LLM）                | 💬 聊天模型（ChatModel / ChatOpenAI）              |
| 构造的 Prompt 类型 | 单个字符串                        | 多个对话消息（message list）                        |
| 支持角色设定       | ❌ 不支持                          | ✅ 支持（System, User, Assistant, Function 等）     |
| 使用格式           | "你是谁？我是{role}。"            | HumanMessagePromptTemplate, SystemMessage...        |
| 适配模型           | `LLM()`                           | `ChatOpenAI()`、`ChatAnthropic()` 等                |

ChatPromptTemplate 常用到 .from_template() 和 .from_messages() 方法，前者接受一个简答字符串作为参数，并返回一个 ChatPromptTemplate 对象，后者则支持消息类型更丰富的组合，如 HumanMessagePromptTemplate、SystemMessagePromptTemplate 等。

## 输出格式解析器（OutputParser）
输出格式解析器（OutputParser）是 LangChain 中用于解析模型输出结果的重要工具。它定义了模型输出结果的格式，并根据格式进行解析，以获取所需的结果（是一种后处理的方法，一般就是把原始的输出结果进行解析，得到想要的结果如 json、list、xml、csv 或自定义等格式）。

值得注意的是，不同解析的格式的调用方法很可能有较大区别，详情请参考 LangChain 有关 [output parser](https://python.langchain.com/api_reference/core/output_parsers.html) 的文档。

## 链（Chain）及其通用接口
链（Chain）是 LangChain 中用于组合多个模块的组件，它们通常用于解决特定的问题或任务。链可以包含多个模块，如 PromptTemplate、LLM、OutputParser 等，它们按照一定的顺序进行组合，以实现特定的功能。

最新版本的 Langchain 中提供了 (LCEL) 语法格式来实现链的组合，其中 `|` 表示链的连接(类似Python的管道符)。比如 `prompt | llm | parser` 就表示一个链，其中 prompt 是一个 PromptTemplate 对象，llm 是一个 LLM 对象，parser 是一个 OutputParser 对象。

链的通用接口包括：
1. `invoke()` 方法：用于执行链，接收一个输入参数，并返回一个输出结果。
2. `batch()`: 高效地处理多个输入，返回一个包含多个输出的列表。
3. `stream()`: 用于流式处理输出，返回一个可迭代对象，可以逐个获取输出结果。