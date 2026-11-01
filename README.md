# LangChain Exploration & Applied LLM Engineering

A comprehensive, hands-on collection of notebooks and API services exploring the **LangChain** ecosystem. This workspace documents the progressive implementation of Large Language Model (LLM) workflows—from foundational model invocations and prompt engineering to **Retrieval-Augmented Generation (RAG)** pipelines, **LangChain Expression Language (LCEL)** orchestration, **conversational memory**, and **REST API deployment** with **LangServe** and **FastAPI**.

---

## What Was Built & Explored

### 1. LLM & Chat Model Interactions
- **Multi-Provider Model Setup**: Configured and compared models across **Groq** (`ChatGroq` running `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `llama3-70b-8192`, and `mixtral-8x7b-32768`) and **OpenAI** (`ChatOpenAI` running `gpt-3.5-turbo` and `gpt-4o-mini`).
- **Execution Modes**: Implemented synchronous generation (`.invoke()`), real-time token streaming (`.stream()`), and parallel batch execution (`.batch()`).
- **Generation Control & Message Schemas**: Tuned model creativity via `temperature` parameters and structured role-based conversations using `SystemMessage`, `HumanMessage`, and tuple-based message definitions.

### 2. Prompt Engineering & Structured Output Parsing
- **Dynamic Prompt Templates**: Built reusable prompts for completion and chat models using `PromptTemplate` and `ChatPromptTemplate`.
- **Few-Shot Prompting**: Guided model behavior and output formatting by injecting example interactions via `FewShotChatMessagePromptTemplate`.
- **Structured Outputs & Validation**: Transformed raw LLM responses into clean strings (`StrOutputParser`), JSON objects (`SimpleJsonOutputParser`), and strictly typed **Pydantic** schemas using `JsonOutputParser` with automated format instructions.

### 3. Document Ingestion & Multi-Format Data Loaders
- Built ingestion workflows capable of extracting text and metadata (`Document` objects) from diverse sources:
  - **Plain Text** (`TextLoader` and native `pathlib` wrapping)
  - **CSV Files** (`CSVLoader`)
  - **HTML Pages** (`UnstructuredHTMLLoader`)
  - **PDF Documents** (`PyPDFLoader` with page-level splitting)
  - **Live Web & Knowledge Bases** (`WikipediaLoader` and `WebBaseLoader` with `BeautifulSoup` parsing)

### 4. Retrieval-Augmented Generation (RAG) Architecture
- **Text Chunking Strategies**: Evaluated document segmentation using `CharacterTextSplitter` and `RecursiveCharacterTextSplitter` with configurable `chunk_size` and `chunk_overlap`.
- **Vector Embeddings**: Generated dense semantic representations locally using HuggingFace Sentence Transformers (`all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`) as well as `OpenAIEmbeddings`.
- **Vector Databases & Retrievers**: Indexed document chunks into **FAISS** and **Chroma** vector stores to perform similarity searches and configured top-$k$ retrievers (`.as_retriever(search_kwargs={"k": ...})`).
- **End-to-End RAG Pipelines**: Connected retrievers, context-formatting functions, prompt templates, and LLMs into grounded Q&A pipelines over custom documents and web articles.

### 5. LangChain Expression Language (LCEL) & Runnables
- **Declarative Pipe Composition**: Replaced legacy `LLMChain` constructs with composable LCEL chains using the `|` operator.
- **Core Runnables**:
  - `RunnablePassthrough`: Forwarding inputs unchanged through pipeline stages.
  - `RunnableLambda`: Wrapping custom Python transformation functions directly inside chains.
  - `RunnableParallel`: Executing multiple branches concurrently and constructing multi-key input dictionaries alongside Python's `itemgetter`.
- **Advanced Chain Patterns**: Used `.bind()` to attach runtime stop sequences (`stop=[...]`) and composed multi-stage nested chains where the output of one LLM chain feeds directly into downstream chains.

### 6. Conversational Memory Management
- Implemented stateful chat architectures capable of retaining context across turns:
  - **Full Buffer Memory**: Storing complete dialogue histories with `ConversationBufferMemory` and injecting them via `MessagesPlaceholder`.
  - **Sliding Window Memory**: Limiting context length to the last $k$ interactions using `ConversationBufferWindowMemory`.
  - **Direct Message History**: Managing explicit user and AI message logs with `ChatMessageHistory`.

### 7. Serving Chains as APIs with LangServe & FastAPI
- Packaged LCEL chains and full FAISS-backed RAG pipelines into production-ready **FastAPI** web servers using **LangServe** (`add_routes`), exposing REST endpoints, OpenAPI documentation, and interactive browser playgrounds.

---

## Repository Structure

```text
├── data/                               # Sample datasets (TXT, PDF, CSV, HTML, SQLite) for RAG & loaders
├── 01_talk_with_llm.ipynb              # LLM initialization, streaming, temperature, and Groq/OpenAI chat models
├── 02_talk_with_llm_part2.ipynb        # PromptTemplates, Few-Shot prompting, and JSON/Pydantic Output Parsers
├── 03_Data_loader_rag_demo.ipynb       # Document loaders (TXT, CSV, HTML, PDF, Wikipedia) & context injection
├── 04_RAG_components.ipynb             # Text splitters, HuggingFace embeddings, Chroma & FAISS vector stores
├── 05_Basic_RAG_Implementation.ipynb   # End-to-end RAG Q&A pipeline built with FAISS, Groq, and LCEL
├── 06_LCEL.ipynb                       # LCEL fundamentals, legacy LLMChain comparison, stream & batch execution
├── 07_LCEL_builtin_runnables.ipynb     # RunnablePassthrough, RunnableLambda, RunnableParallel & itemgetter
├── 08_LCEL_little_more.ipynb           # Runtime .bind(), multi-chain composition, and web-based RAG with Chroma
├── 09_Memory_langchain.ipynb           # ConversationBufferMemory, WindowMemory (k-window), & ChatMessageHistory
├── 10_langserve_demo.py                # FastAPI + LangServe microservice for multi-language translation
├── 11_rag_langserve_demo.py            # FastAPI + LangServe microservice serving a full FAISS RAG pipeline
├── .env.example                        # Template for required environment variables (API keys)
├── .gitignore                          # Git exclusion rules for secrets, virtualenvs, and caches
└── requirements.txt                    # Python package dependencies
```

### Breakdown by Module

| File | Focus Area | Key Components Used |
| :--- | :--- | :--- |
| `01_talk_with_llm.ipynb` | **Foundational LLM Calls** | `ChatGroq`, `ChatOpenAI`, `.invoke()`, `.stream()`, `SystemMessage`, `HumanMessage` |
| `02_talk_with_llm_part2.ipynb` | **Prompts & Output Parsers** | `PromptTemplate`, `ChatPromptTemplate`, `FewShotChatMessagePromptTemplate`, `SimpleJsonOutputParser`, `JsonOutputParser`, `Pydantic` |
| `03_Data_loader_rag_demo.ipynb` | **Data Ingestion** | `TextLoader`, `CSVLoader`, `UnstructuredHTMLLoader`, `PyPDFLoader`, `WikipediaLoader` |
| `04_RAG_components.ipynb` | **RAG Building Blocks** | `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `Chroma`, `FAISS` |
| `05_Basic_RAG_Implementation.ipynb` | **Complete RAG Pipeline** | `FAISS.as_retriever()`, `RunnablePassthrough`, `ChatPromptTemplate`, `StrOutputParser` |
| `06_LCEL.ipynb` | **LCEL Core Mechanics** | Pipe (`\|`) syntax, `LLMChain` comparison, `.invoke()`, `.stream()`, `.batch()` |
| `07_LCEL_builtin_runnables.ipynb` | **Built-in Runnables** | `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel`, `operator.itemgetter` |
| `08_LCEL_little_more.ipynb` | **Advanced LCEL & Chaining** | `.bind(stop=...)`, multi-step chain composition, `WebBaseLoader` + `Chroma` RAG |
| `09_Memory_langchain.ipynb` | **Conversational Memory** | `ConversationBufferMemory`, `ConversationBufferWindowMemory`, `ChatMessageHistory`, `MessagesPlaceholder` |
| `10_langserve_demo.py` | **Translation API Server** | `FastAPI`, `langserve.add_routes`, `uvicorn`, `ChatGroq` |
| `11_rag_langserve_demo.py` | **RAG Q&A API Server** | `FastAPI`, `langserve.add_routes`, `FAISS`, `HuggingFaceEmbeddings`, `RecursiveCharacterTextSplitter` |

---

## Tech Stack

- **Orchestration Framework**: `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`
- **LLM Providers**: `langchain-groq` (Qwen, LLaMA 3, Mixtral, GPT-OSS), `langchain-openai` (GPT-3.5-Turbo, GPT-4o-mini)
- **Embeddings**: `langchain-huggingface`, `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Stores**: `faiss-cpu`, `langchain-chroma`
- **Document Parsing**: `pypdf`, `wikipedia`, `beautifulsoup4`
- **API & Deployment**: `langserve`, `fastapi`, `uvicorn`, `sse-starlette`
- **Environment & Validation**: `python-dotenv`, `pydantic`

---

## Getting Started

### 1. Install Dependencies

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` in the project root directory and add your API keys:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Notebooks

Open any of the numbered notebooks (`01` through `09`) in Jupyter Lab, Jupyter Notebook, or VS Code to step through the implementations interactively.

### 4. Run the LangServe API Servers

#### Translation API (`10_langserve_demo.py`)
```bash
python 10_langserve_demo.py
```
- **API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Interactive Playground**: `http://localhost:8000/chain/playground/`

#### RAG Document Q&A API (`11_rag_langserve_demo.py`)
```bash
python 11_rag_langserve_demo.py
```
- **API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Interactive Playground**: `http://localhost:8000/rag/playground/`

