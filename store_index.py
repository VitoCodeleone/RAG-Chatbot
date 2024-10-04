from src.helper import load_pdf, text_split, download_hf_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

#extracting PDF data and splitting it into chunks
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embedding = download_hf_embeddings()

#initialising pinecone
index_name = "medical-chatbot"
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embedding, index_name = index_name)
