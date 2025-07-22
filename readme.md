# 📄 RAG System with Hugging Face LLM & Streamlit

This project is a fully functional Retrieval-Augmented Generation (RAG) system that allows users to:

1. **Upload a PDF document**
2. **Split and embed it**
3. **Store and search the vectorized chunks**
4. **Ask questions about the document**
5. **Get answers using a free Hugging Face LLM**

## LLM Used
No OpenAI key required — 100% free using `google/flan-t5-base`!

---

##  Features

-  Uses `sentence-transformers` for generating document embeddings
-  Fast similarity search using **FAISS**
-  Free LLM (`flan-t5-base`) from Hugging Face Transformers
-  Supports PDF document ingestion
-  Simple and clean **Streamlit frontend**
-  Built using LangChain for composability


##  How to Run

### Step 1: Setup & Run the Backend in Colab
- Open rag_backend_colab.ipynb in Google Colab

- Upload your PDF and run all cells

- (Optional) Save the faiss_index folder if you want to persist vectorstore

### Step 2: Run Frontend Locally
```
streamlit run streamlit_app.py
```

- Upload a PDF

- Click "Process Document"

- Ask questions like “What is the document about?”