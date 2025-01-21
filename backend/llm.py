from uuid import uuid4

from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()


# Initialize OpenAI LLM
def load_llm(model_name: str = "phi3", temperature=0.3):
    return OllamaLLM(model=model_name, temperature=temperature)


# Initialize Vector Store
def create_vectorstore(persist_directory: str = "chroma_db"):
    embeddings = OllamaEmbeddings(model="phi3")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Combine LLM and Vector Store into QA Chain
def create_qa_chain(llm, vectorstore, k=3):
    prompt = hub.pull("rlm/rag-prompt")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


# Helper function to process and add documents
def process_and_add_documents(
    vectorstore, documents, chunk_size=500, chunk_overlap=100
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vectorstore.add_documents(documents=chunks, ids=uuids)
    vectorstore.persist()
