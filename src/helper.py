from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import re
import uuid
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

pc = Pinecone(api_key=PINECONE_API_KEY)

#Extract data from pdf
def load_pdf(path):
    loader = PyPDFLoader(path)

    pages = []

    for page in loader.lazy_load():
        pages.append(page)
    
    return pages
    

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

#Download HF embedding model
def download_hf_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

#create index name
def create_index_name(pdf):
    base_name = pdf.rsplit('.', 1)[0]
    sanitized_name = re.sub(r'[^a-z0-9]+', '-', base_name.lower()).strip('-')
    unique_id = str(uuid.uuid4())[:8]
    return f"{sanitized_name}-{unique_id}"

#store_index
def store_index(upload_dir, filename, embedding):
    path = os.path.join(upload_dir, filename)
    extracted_data = load_pdf(path)
    texts = text_split(extracted_data)
    index_name = create_index_name(filename)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            metric="cosine",
            dimension=384,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding)
        vector_store.add_texts([t.page_content for t in texts])
    
    return index_name

#create_retriever
def create_retriever(index_name, embedding):
    try:
        docsearch = PineconeVectorStore(index_name=index_name, embedding=embedding)
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        return retriever
    except:
        print("Error in creating retriever")

#create rag chain
def create_rag_chain(retriever, prompt):
    try:
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

        return rag_chain
    except:
        print("Error in creating rag chain")