import os
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Fix for "no current event loop" error
nest_asyncio.apply()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Load and split documents from JSON
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",        # Read each element in array
    text_content=False
)
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Create or load FAISS index
if os.path.exists("faiss_index"):
    db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
else:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

# Retrieval chain with higher top_k so we don't miss results
retriever = db.as_retriever(search_kwargs={"k": 15})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("RAG App with LangChain + Gemini + FAISS")
query = st.text_input("Ask a question from the JSON document")
if query:
    try:
        answer = qa.run(query)  # sync safe
        st.write("**Answer:**", answer)
    except Exception as e:
        st.error(f"Error: {e}")
