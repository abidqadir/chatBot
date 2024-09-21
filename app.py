
from flask import Flask ,render_template ,request
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader 
import os


app = Flask(__name__)

GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    if request.method == 'POST':
        link = request.form['website_link']
        query = request.form['query']
        
        try:
            loader = WebBaseLoader(link)
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

            response = qa_chain.invoke({"query": query})  # Use the user-defined query
            answer = response['result']  # Adjust as necessary based on your response structure

        except Exception as e:
            answer = f"An error occurred: {str(e)}"
    
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)