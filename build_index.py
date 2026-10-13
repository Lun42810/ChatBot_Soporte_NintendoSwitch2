import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from PyPDF2 import PdfReader

# Carpeta donde están tus manuales
PDF_FOLDER = "documents"
INDEX_PATH = "data/vector_index"

# Modelo de embeddings (rápido y preciso)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
