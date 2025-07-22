import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
import tempfile, os
import torch

# LLM
@st.cache_resource
def load_hf_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )
    return HuggingFacePipeline(pipeline=pipe)

# Load vector store from uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# QA Chain
def ask_question(vectorstore, question):
    llm = load_hf_pipeline()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")
    return qa.run(question)

# UI
st.set_page_config(page_title="RAG - Local 🤖", layout="centered")
st.title("📄 Ask Questions from Uploaded PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("Uploaded successfully. Click Process to continue.")
    if st.button("Process Document"):
        st.session_state['vectorstore'] = process_pdf(uploaded_file)
        st.success("Document processed!")

if 'vectorstore' in st.session_state:
    question = st.text_input("Ask your question")
    if question:
        with st.spinner("Thinking..."):
            response = ask_question(st.session_state['vectorstore'], question)
        st.markdown(f"**Answer:** {response}")
