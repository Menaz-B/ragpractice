import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="PDF Q&A with GPT", layout="wide")
st.title("📄 Chat with Your PDF using GPT + FAISS")

openai_api_key = st.secrets["general"]["openai_api_key"]

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and openai_api_key:
    # Read PDF
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(raw_text)

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)

    # Build QA chain
    llm = OpenAI(openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # User query
    query = st.text_input("Ask something about your PDF:")
    if query:
        with st.spinner("Thinking..."):
            result = qa.run(query)
            st.write(result)
