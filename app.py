from flask import Flask, render_template, jsonify, request
from src.helper import download_hf_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

embedding = download_hf_embeddings()

index_name = "medical-chatbot"
docsearch = Pinecone.from_existing_index(index_name, embedding)

PROMPT = PromptTemplate(template=prompt_template, input_variables = ["context", "question"])

model = OllamaLLM(model='llama3.1')

qa=RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

@app.route("/")
@app.route("/home")
def home():
    return render_template("chat.html", title="Chatbot")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=True)