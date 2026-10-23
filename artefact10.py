# Install required packages (run once in your terminal, not in the script):
# pip install openai numpy langchain langchain_classic langchain-community langchain-openai
#             langchain-chroma langchain-text-splitters chromadb
#             llama-index llama-index-embeddings-azure-openai

# Set environment variables before running any script:
#   macOS / Linux:        export API_KEY=your_key_here
#   Windows (PowerShell): $env:API_KEY = "your_key_here"

# The 'data/' folder should contain the .txt lecture transcript files. Place it in the same directory as your scripts.
# Label each block A–F with the RAG module it implements:
# Ingestion | Chunking | Embedding | Vector Store | Retrieval | Generation
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

APIM_CHAT_URL       = "https://qut-ai.azure-api.net/ifb220/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-03-01-preview"
APIM_EMBEDDINGS_URL = "https://qut-ai.azure-api.net/ifb220/openai/deployments/text-embedding-3-small/embeddings?api-version=2025-03-01-preview"
API_KEY             = os.environ["API_KEY"]

# ── Block A ───────────────────────────────────────────────────────────────────
loader    = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
print(f"Block A: loaded {len(documents)} document(s).") # Ingestion - loads documents from directory using loaders.

# ── Block B ───────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks   = splitter.split_documents(documents)
print(f"Block B: produced {len(chunks)} chunks.") # Chunking - splits documents into chunks.

# ── Block C ───────────────────────────────────────────────────────────────────
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=APIM_EMBEDDINGS_URL,
    azure_deployment="text-embedding-3-small",
    api_key=API_KEY,
    api_version="2025-03-01-preview" # Embedding - instantiate embeddings model but not used yet. 
                                     # However it's just creating the embedding model, so it's preparing embedding module? 
                                     # It instantiates model which will be used later. Might count as Embedding.
)

# ── Block D ───────────────────────────────────────────────────────────────────
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db_ifb220"
)
print(f"Block D: {db._collection.count()} chunk embeddings stored.")
# Vector Store - stores embeddings to Chroma DB. 
# Actually it uses embedding model to compute embeddings, and stores into vector store. 
# So this block implements Vector Store (and maybe embedding since embeddings are computed). 
# But typical mapping: D implements Vector Store. 
# It also uses the embeddings, but that is part of embedding module? 
# The block itself is storing vector.


# ── Block E ───────────────────────────────────────────────────────────────────
retriever = db.as_retriever(search_kwargs={"k": 3})
# Retrieval - sets up retriever from db.

# ── Block F ───────────────────────────────────────────────────────────────────

# Generation - sets up LLM and the RetrievalQA chain.

llm = AzureChatOpenAI(
    azure_endpoint=APIM_CHAT_URL,
    azure_deployment="gpt-4.1-mini",
    api_key=API_KEY,
    api_version="2025-03-01-preview"
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ── Test questions — record results in the table below ────────────────────────
questions = [
    ("direct",      "What is the main purpose of the chunking module in a RAG pipeline?"),
    ("synthesis",   "What are the main differences between LangChain and LlamaIndex for RAG?"),
    ("direct",      "What are the portfolio submission requirements for Artefact 6?"),
    ("out-of-scope","What is the current interest rate set by the Reserve Bank of Australia?"),
]

for q_type, question in questions:
    result = qa_chain.invoke({"query": question})
    print(f"\n[{q_type.upper()}] {question}")
    print(f"Answer: {result['result']}")
    print(f"Chunks retrieved: {len(result['source_documents'])}")
    for i, doc in enumerate(result['source_documents']):
        print(f"  Chunk {i+1}: {doc.page_content[:100].strip()}...")