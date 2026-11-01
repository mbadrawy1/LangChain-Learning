import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["GROQ_API_KEY"]

# 1. Load Document
file_path = "./data/state_of_the_union.txt"
text = Path(file_path).read_text(encoding="utf-8")
loaded_document = [Document(page_content=text, metadata={"source": file_path})]

# 2. Split Document into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks_of_text = text_splitter.split_documents(loaded_document)

# 3. Create Embeddings & Vector Store (FAISS)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks_of_text, embeddings_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 4. Initialize LLM, Prompt, and Output Parser
llm = ChatGroq(model="qwen/qwen3.8-27b")
parser = StrOutputParser()

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# 5. Build RAG Chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# 6. Create FastAPI App & Add LangServe Routes
app = FastAPI(
    title="RAG Document Q&A Server",
    version="1.0",
    description="A Retrieval-Augmented Generation (RAG) API server using LangChain, FAISS, and LangServe",
)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

