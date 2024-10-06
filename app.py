from flask import Flask, render_template, jsonify, request, redirect, flash, url_for
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from src.helper import download_hf_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
from src.forms import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

app.config["SECRET_KEY"] = os.getenv('SECRET_KEY')
app.config["MONGO_URI"] = os.getenv('MONGO_URI')

db = PyMongo(app).db

bcrypt = Bcrypt(app)

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

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    login = LoginForm()
    signup = SignUpForm()
    return render_template("home.html", title="Home", login=login, signup=signup), 200

@app.route("/chat")
def chat():
    return render_template("chat.html", title="Chatbot")

@app.route("/reply", methods=["GET", "POST"])
def reply():
    try:
        msg = request.get_json()
        input = msg["msg"]
        result = ''.join(rag_chain.stream(input))
        return str(result), 200
    except:
        return 'ServerError: Please try again later'
    
@app.route("/login", methods=['GET', 'POST'])
def login():
    login_form = LoginForm()
    signup_form = SignUpForm()
    if login_form.validate_on_submit():
        email = signup_form.email.data
        password = signup_form.password.data

        user = db.users.find_one({"email": email})

        if user:
            if bcrypt.check_password_hash(user['password'], password):
                return redirect(url_for("dashboard"))
            
            else:
                flash("Incorrect password. Please try again", "password-error")
        else:
            flash("Email address does not exist", "email-not-exists")

    return render_template("home.html", title="Home", login=login_form, signup=signup_form)

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    login_form = LoginForm()
    signup_form = SignUpForm()
    if signup_form.validate_on_submit():
        email = signup_form.email.data
        password = signup_form.password.data

        if db.users.find_one({"email": email}):
            flash("Email address already exists", "email-exists")
            
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            db.users.insert_one({
                "email": email,
                "password": hashed_password
            })

            return redirect(url_for("dashboard"))

    return render_template("home.html", title="Home", login=login_form, signup=signup_form, show_signup=True)

@app.route("/dashboard")
def dashboard():
    upload_form = UploadForm()
    return render_template("dashboard.html", title="Dashboard", upload=upload_form)

@app.route("/pdf-upload", methods=["GET", "POST"])
def upload():
    upload_form = UploadForm()
    if upload_form.validate_on_submit():
        file = upload_form.pdf.data
        file.save(os.path.join("./data", secure_filename(file.filename)))
        return redirect(url_for("chat"))
    
    return render_template("dashboard.html", title="Dashboard", upload=upload_form)

if __name__ == "__main__":
    app.run(debug=True)