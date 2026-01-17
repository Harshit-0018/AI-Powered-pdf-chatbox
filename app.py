import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot (RAG)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add it in Streamlit Secrets.")
    st.stop()

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vectorstore
try:
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
except:
    st.error("Vectorstore not found. Make sure 'vectorstore/' is pushed to GitHub.")
    st.stop()

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

query = st.text_input("Ask a question from your PDF")

if query:
    docs = db.similarity_search(query, k=1)
    context = docs[0].page_content

    response = llm(
        context + "\n\nQuestion: " + query
    )

    st.write("### Answer")
    st.success(response)
