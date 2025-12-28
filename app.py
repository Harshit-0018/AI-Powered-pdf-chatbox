
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot (RAG)")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

llm = OpenAI(temperature=0)

query = st.text_input("Ask a question from your PDF")

if query:
    docs = db.similarity_search(query)
    context = docs[0].page_content
    response = llm(context + "\nQuestion: " + query)
    st.write("### Answer")
    st.success(response)
