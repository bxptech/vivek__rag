import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    JSONLoader
)

load_dotenv()
#Load documents from various formats
all_docs = []
# #Load PDF
# pdf_path = "data/harry potter.pdf"
# if os.path.exists(pdf_path):
#     pdf_loader = PyPDFLoader(pdf_path)
#     all_docs.extend(pdf_loader.load())

# #Load DOCX
# docx_path = "data/sample.docx"
# if os.path.exists(docx_path):
#     docx_loader = UnstructuredWordDocumentLoader(docx_path)
#     all_docs.extend(docx_loader.load())

# #Load Excel
# excel_path = "data/sample.xlsx"
# if os.path.exists(excel_path):
#     excel_loader = UnstructuredExcelLoader(excel_path)
#     all_docs.extend(excel_loader.load())

#Load JSON
json_path = "data/data.json"
if os.path.exists(json_path):
    json_loader = JSONLoader(
        file_path=json_path,
        jq_schema=".[]",
        text_content=False,
        json_lines=False,
    )
    all_docs.extend(json_loader.load())

#Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

#Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

#Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

#Create RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#Ask a query
query ="""get me report name  for report fields  ["Sup.Bill Date|Sup.Bill No", "Gst No", "HSN Code", "Pack", "Qty", "Rate", "Total", "Discount", "Gross", "CGST", "SGST", "IGST"]											
"""""
result = qa_chain.run(query)
print("Answer:", result)
