from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.indexes import SQLRecordManager, index

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

# 加载 llm 模型
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
    collection_name="collection3",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),#这里size的768是根据模型的embedding向量维度来设置的，两者一致
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="collection3",
    embedding=embeddings,
)
# step 5: 将 document 通过 huggingface 的 embeddings 对象计算 embedding 向量信息并临时存入 Qdrant 向量数据库，用于后续匹配查询
vector_store.add_documents(split_docs)
# step 6: 创建问答对象
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True)
# step 7: 进行问答
# result = qa.invoke({"query": "三家分晋是哪三家？"})
# print(result['result'])
# print(result['source_documents'])

# 初始化记录管理器
record_manager = SQLRecordManager(namespace="my_namespace", db_url="sqlite:///records.db")
# 创建数据库模式
record_manager.create_schema()

# 索引文档
indexs = index(
    docs_source=split_docs,
    record_manager=record_manager,
    vector_store=vector_store,
    cleanup="incremental",
    source_id_key="source"
)
print(indexs)
# indexs.index(documents)