import streamlit as st
import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).parent.parent

@st.cache_resource
def setup_rag_pipeline():
    """Sets up the RAG pipeline: loads documents, chunks them, embeds, and creates the vector store/retriever."""
    DATA_PATH = BASE_DIR / "it_docs"
    CHROMA_PATH = BASE_DIR / "chroma_db"

    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        st.warning(f"No documents found in: {DATA_PATH}")
        return None

    #Load Documents
    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.txt")
    raw_docs = loader.load()

    #Split Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH)
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    st.success(f"Knowledge Base Ready — {len(docs)} chunks loaded.")
    return retriever

def combine_documents(docs):
    """Combines a list of documents into a single string."""
    return "\n\n".join(d.page_content for d in docs)