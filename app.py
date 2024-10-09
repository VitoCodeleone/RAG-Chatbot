from flask import Flask, render_template, jsonify, request, redirect, flash, url_for, session, make_response
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from src.helper import download_hf_embeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from src.prompt import *
from src.forms import *
from src.helper import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

app.config["SECRET_KEY"] = os.getenv('SECRET_KEY')
app.config["MONGO_URI"] = os.getenv('MONGO_URI')
app.config["UPLOAD_PATH"] = "./data"

db = PyMongo(app).db

bcrypt = Bcrypt(app)

embedding = download_hf_embeddings()


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    login = LoginForm()
    signup = SignUpForm()
    return render_template("home.html", title="Home", login=login, signup=signup), 200

@app.route("/chat")
def chat():
    if "email" not in session:
        flash("Please login first!", "login-required")
        return redirect(url_for("home", next=request.url))
    
    return render_template("chat.html", title="Chatbot", email=session["email"])

@app.route("/reply", methods=["GET", "POST"])
def reply():
    try:
        msg = request.get_json()
        input = msg["msg"]
        retriever = create_retriever(session["index_name"], embedding)
        rag_chain = create_rag_chain(retriever, prompt)
        result = ''.join(rag_chain.stream(input))
        return str(result), 200
    except:
        return 'ServerError: Please try again later', 500
    
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
                session["email"] = email
                next_url = request.args.get("next")
                return redirect(next_url or url_for("dashboard"))
            
            else:
                flash("Incorrect password. Please try again", "password-error")
        else:
            flash("Email address does not exist", "email-not-exists")

    return render_template("home.html", title="Home", login=login_form, signup=signup_form), 400

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

            session["email"] = email
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))

    return render_template("home.html", title="Home", login=login_form, signup=signup_form, show_signup=True), 400

@app.route("/logout")
def logout():
    session.pop('email', None)
    redir = make_response(redirect(url_for("home")))
    redir.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return redir

@app.route("/dashboard")
def dashboard():
    if "email" not in session:
        flash("Please login first!", "login-required")
        return redirect(url_for("home", next=request.url))

    upload_form = UploadForm()
    return render_template("dashboard.html", title="Dashboard", upload=upload_form, email=session["email"]), 200

@app.route("/pdf-upload", methods=["GET", "POST"])
def upload():
    if "email" not in session:
        flash("Please login first!", "login-required")
        return redirect(url_for("home", next=request.url))

    upload_form = UploadForm()
    if upload_form.validate_on_submit():
        file = upload_form.pdf.data
        filename = secure_filename(file.filename)
        
        if filename[-3:] != "pdf":
            flash("Supported file types: .pdf", "filetype-error")
        elif len(file.read()) > 20 * 1024 * 1024:
            flash("File size should be less than 20MB", "too-large")
        else:
            file.seek(0)
            file.save(os.path.join(app.config["UPLOAD_PATH"], filename))
            session["index_name"] = store_index(app.config["UPLOAD_PATH"], filename, embedding)
            return redirect(url_for("chat"))
    
    return render_template("dashboard.html", title="Dashboard", upload=upload_form, email=session["email"]), 400

if __name__ == "__main__":
    app.run(debug=True)