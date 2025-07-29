import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import SimpleJSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Load and split documents
loader = SimpleJSONLoader("data.json")  # Your JSON file path
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Create or load FAISS index
if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

# Retrieval chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.title("RAG App with LangChain + Gemini + FAISS")
query = st.text_input("Ask a question from the JSON document")
if query:
    answer = qa.run(query)
    st.write("**Answer:**", answer)
