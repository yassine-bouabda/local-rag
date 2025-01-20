
from uuid import uuid4
import os
import openai
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub 
from langchain_ollama import OllamaLLM

from langchain_ollama import OllamaEmbeddings



# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")
# Set OpenAI API key from environment variable (ensure your OpenAI key is set in your environment)


# Initialize OpenAI LLM
def load_llm(model_name: str = "phi3",temperature=0.3):
    return OllamaLLM(model=model_name, temperature=temperature)

# Initialize Vector Store
def create_vectorstore(persist_directory: str = "chroma_db"):
    embeddings = OllamaEmbeddings(
    model="phi3")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Combine LLM and Vector Store into QA Chain
def create_qa_chain(llm, vectorstore,k=3):
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
def process_and_add_documents(vectorstore, documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vectorstore.add_documents(documents=chunks, ids=uuids)
    vectorstore.persist()
