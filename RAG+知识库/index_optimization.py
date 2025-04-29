from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore # 使用内存作为 Docstore 的示例
import uuid # 用于生成唯一 ID

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')

# step 0:加载 llm 模型
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key # type: ignore
    model="qwen-plus",	# 模型名称
    temperature=0,
)
# step 1: 递归查找加载目录中的所有txt类型的文件
loader = DirectoryLoader('./RAG+知识库/txt_files/', glob='**/*.txt',loader_cls=TextLoader)
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
# step 2: 文本拆分
# 初始化文本加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)
# step 3: 利用embeddings将导入的文档数据转成向量
embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')

# step 4: 创建Qdrant向量数据库
# client = QdrantClient(":memory:")#采用内存数据库，程序完了就没了；此外还可以选择本地磁盘存储，云端存储
client = QdrantClient(path="./RAG+知识库/langqdrant")#本地磁盘存储
client.create_collection(
    collection_name="collection_index_optimization",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),#这里size的768是根据模型的embedding向量维度来设置的，两者一致
)

# 初始化 Docstore 用于存储原始文档块
# 使用 InMemoryStore 作为简单的示例 Docstore
docstore = InMemoryStore()

# 生成唯一 ID，并存储原始文档块到 Docstore
doc_ids = [str(uuid.uuid4()) for _ in split_docs] # 为每个分割后的文档块生成唯一 ID
docstore.mset(list(zip(doc_ids, split_docs))) # 将 ID 和对应的原始文档块存储到 Docstore

summaries = []
for i, doc in enumerate(split_docs):
    # 构建生成摘要的提示
    prompt = f"请为以下文档块生成一个简洁的摘要，用于后续快速检索：\n\n{doc.page_content}\n\n摘要："
    # 调用 LLM 生成摘要
    summary_response = llm.invoke(prompt)
    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    summaries.append(summary)
    print(f"已处理文档块 {i+1}/{len(split_docs)}")

# 将摘要嵌入，并存储到**专门用于摘要的 Qdrant 集合**中
# 注意：这里我们使用前面定义的 embeddings 模型
summary_vectorstore = QdrantVectorStore(
    client=client,
    collection_name="collection_index_optimization",
    embedding=embeddings, # 使用相同的 embedding 模型
)
# 将摘要添加到向量存储，并关联原始文档块的 ID
# 这里我们将原始文档块的 ID 存储在摘要的 metadata 中
summary_vectorstore.add_texts(
    summaries,
    metadatas=[{"doc_id": doc_ids[i]} for i in range(len(summaries))]
    # 确保这里的索引 i 是正确的，对应 summaries 和 doc_ids
)
# step 6: 创建 MultiVectorRetriever 实例
# 配置 MultiVectorRetriever 指向摘要的向量存储和原始文档块的 Docstore
multivector_retriever = MultiVectorRetriever(
    vectorstore=summary_vectorstore, # 指向存储摘要嵌入的向量存储
    docstore=docstore,               # 指向存储原始文档块的 Docstore
    id_key="doc_id"                  # 指明在向量存储的 metadata 中使用哪个 key 来查找 Docstore 中的原始文档
)
# 示例查询
query = "请问这些文档主要讲了什么？" # 用户问题

print(f"\n使用 MultiVectorRetriever 检索问题: '{query}'")
retrieved_original_docs = multivector_retriever.invoke(query)

print("\n检索到的原始文档块内容：")
for i, doc in enumerate(retrieved_original_docs):
    print(f"--- 检索到的文档块 {i+1} ---")
    print(doc.page_content[:200] + "...") # 只打印前200个字符
    print("-" * 20)
