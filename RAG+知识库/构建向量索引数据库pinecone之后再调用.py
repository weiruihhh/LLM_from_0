import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
qwen_api_key = os.getenv('QWEN_API_KEY')
# 加载 llm 模型
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,	# app_key
    model="qwen-plus",	# 模型名称
    temperature=0,
)
from pinecone import Pinecone  # type: ignore
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")

index_name = "langchain-database"
embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')

# 递归查找加载目录中的所有txt类型的文件
loader = DirectoryLoader('./知识库/txt_files/', glob='**/*.txt',loader_cls=TextLoader)
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

query = "越王勾践最终灭掉了哪个国家？"
res = vectorstore.similarity_search(query)
print(res)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    split_docs,
    index_name=index_name,
    embedding=embeddings
)
vectorstore_from_texts = PineconeVectorStore.from_texts(
    query,
    index_name=index_name,
    embedding=embeddings
)
# print(vectorstore_from_texts)
