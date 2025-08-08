import os
import streamlit as st
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from euriai.langchain import EuriaiEmbeddings, EuriaiLLM

load_dotenv()  # Load environment variables from .env file

EURIAI_API_KEY = os.getenv("EURIAI_API_KEY")

@st.cache_resource
def load_qa_chain():
    loader = PyPDFLoader("./data/Pavan_Bairu_Resume.pdf")
    documents = loader.load()
    print("Loaded docs:", len(documents))

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print("Text chunks: \n", chunks)

    vector_store = Chroma.from_documents(chunks, EuriaiEmbeddings(api_key=EURIAI_API_KEY))
    retriever = vector_store.as_retriever()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a highly intelligent resume parsing assistant.

Given the resume content below:
{context}

Your task is to:
- Carefully read the question and extract only the **relevant** information.
- Focus on **specific sections** mentioned in the question (e.g., TCS, Capgemini, Projects).
- If the question is about **personal details**, extract them clearly.
- If the question is about **skills**, list them in a structured format.
- If the question is **multi-part**, extract and structure all parts clearly.
- If the question is about **education**, provide details in a structured format.
- If the question is about **experience**, provide details in a structured format.
- If the question is about **projects**, provide details in a structured format.
- If the question is about **contact information**, provide details in a structured format.
- If the question is about **address**, provide details in a structured format.
- If the question is about **certifications**, provide details in a structured format.
- just provide only the **relevant information** in the response.
- Always respond in a **valid Python dictionary** format.
- Use **nested dictionaries or lists** if necessary.
- If the information is not found, return an empty dictionary: {{}}

Now answer the following question:
"{question}"
"""
    )

    llm = EuriaiLLM(
        api_key=EURIAI_API_KEY,
        model="gpt-4.1-nano",
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


qa_chain = load_qa_chain()

# Set page configuration
st.set_page_config(page_title="🧠 RAG Resume Chatbot", layout="wide")

# Centered title and subtitle using HTML
st.markdown(
    "<h1 style='text-align: center;'>🧠 RAG Resume Chatbot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Ask questions about the resume to get structured responses in dictionary format.</p>",
    unsafe_allow_html=True
)

# Full-width question input
user_query = st.text_input(
    label="Ask questions about the resume to get structured responses in dictionary format.",
    placeholder="e.g., What are the skills Pavan has?",
    label_visibility="collapsed"
)

# Query handling
if user_query:
    with st.spinner("Fetching response..."):
        result = qa_chain.invoke({"query": user_query})

    st.markdown("#### 📄 Response:")
    
    # Try parsing and beautifying JSON
    try:
        parsed = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
        st.json(parsed)
    except Exception:
        st.write(result["result"])

