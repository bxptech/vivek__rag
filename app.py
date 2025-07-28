import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import JSONLoader

# Load environment variables (API keys)
load_dotenv()

st.set_page_config(page_title="RAG Report Finder", layout="centered")
st.title("📄 RAG-based Report Finder using Gemini")

@st.cache_resource(show_spinner="🔄 Loading vector store...")
def build_retriever():
    all_docs = []

    # Load JSON
    json_path = "data/data.json"
    if os.path.exists(json_path):
        json_loader = JSONLoader(
            file_path=json_path,
            jq_schema=".[]",
            text_content=False,
            json_lines=False,
        )
        all_docs.extend(json_loader.load())

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Store in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

retriever = build_retriever()

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# User input form
with st.form("query_form"):
    query = st.text_area("🔍 Enter your report field query", 
        placeholder='e.g. ["HSN Code", "Qty", "Rate", "Discount"]')
    submitted = st.form_submit_button("Submit")

if submitted and query.strip():
    with st.spinner("💡 Getting your answer..."):
        try:
            answer = qa_chain.run(query)
            st.markdown("### ✅ Gemini Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"❌ Error: {e}")
