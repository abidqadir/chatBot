import os

from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader 
import os


GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

loader = WebBaseLoader("https://www.xevensolutions.com/")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=100
)

text_chunks = text_splitter.split_documents(documents)

persist_directory = "doc_db"

embedding = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory=persist_directory
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)