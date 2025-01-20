import arxiv
from fastapi import FastAPI, Form, UploadFile,Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from llm import (create_qa_chain, create_vectorstore, load_llm,
                 process_and_add_documents)
from enums import SortCriterion

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LLM, Vector Store, and QA Chain
llm=load_llm()
vectorstore = create_vectorstore()
qa_chain = create_qa_chain(
    llm,
    vectorstore,
)



# Helper: Fetch Most relevant AI papers from ArXiv
@app.get("/fetch_arxiv")
async def fetch_arxiv_papers(
    query: str, criteria:str, max_results: int = 5
):
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
    pdf_loader = PyPDFLoader(file.file)
    documents = pdf_loader.load()
    process_and_add_documents(vectorstore, documents)
    return {"message": "PDF uploaded and processed successfully."}


# Endpoint: Query the assistant
@app.post("/query")
async def query_research_assistant(question: str = Form(...)):
    response = qa_chain.invoke(question)
    return {"response": response}
