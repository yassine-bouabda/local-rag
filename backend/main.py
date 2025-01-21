import os
import tempfile

import arxiv
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader

from enums import SortCriterion
from llm import create_qa_chain, create_vectorstore, load_llm, process_and_add_documents

front_url = os.getenv("FRONTEND_URL")
ollama_url = os.getenv("OLLAMA_BASE_URL")
local_url = "http://backend:8000"

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # [front_url, local_url, ollama_url],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Load LLM, Vector Store, and QA Chain
llm = load_llm()
vectorstore = create_vectorstore()
qa_chain = create_qa_chain(
    llm,
    vectorstore,
)


# Helper: Fetch Most relevant AI papers from ArXiv
@app.get("/fetch_arxiv")
async def fetch_arxiv_papers(query: str, criteria: str, max_results: int = 5):
    criteria = SortCriterion[criteria]
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=criteria,
    )
    papers = []
    for result in search.results():
        papers.append(
            {
                "title": result.title,
                "summary": result.summary,
                "url": result.pdf_url,
                "published": result.published.strftime("%Y-%m-%d"),
            }
        )
    return {"papers": papers}


# Endpoint: Upload PDF
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Write the uploaded file to the temporary file
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        # Use the file path with PyPDFLoader
        pdf_loader = PyPDFLoader(temp_file_path)
        documents = pdf_loader.load()
        process_and_add_documents(vectorstore, documents)
        return {"message": "PDF uploaded and processed successfully."}
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Endpoint: Query the assistant
@app.post("/query")
async def query_research_assistant(question: str = Form(...)):
    response = qa_chain.invoke(question)
    return {"response": response}
