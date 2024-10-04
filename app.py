from flask import Flask, render_template, jsonify, request
from src.helper import download_hf_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

embedding = download_hf_embeddings()

index_name = "medical-chatbot"
docsearch = Pinecone.from_existing_index(index_name, embedding)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})


local_llm = 'llama3.2'
llm = ChatOllama(model=local_llm)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
@app.route("/home")
def home():
    return render_template("chat.html", title="Chatbot")

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.get_json()
        input = msg["msg"]
        result = ''.join(rag_chain.stream(input))
        return str(result)
    except:
        return 'ServerError: Please try again later'

if __name__ == "__main__":
    app.run(debug=True)