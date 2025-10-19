import os
import gradio as gr
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = "dataset_1/"
PERSIST_DIRECTORY = "chroma_db"
LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert chatbot specializing in comparing the economies of developing and developed countries. 
You have deep knowledge of economic indicators, development patterns, and policy implications. 
When answering questions, don't purely rely on retrieved context—use your pretraining and economic knowledge to make informed conclusions and connections. 
Provide nuanced, thoughtful analysis that goes beyond just regurgitating data."""

# Global variable
rag_chain = None

def load_documents():
    print("\n[LOAD DOCS] Checking dataset path...")
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset directory '{DATA_PATH}' not found")
    
    print(f"[LOAD DOCS] Path exists. Searching for .txt files in {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No .txt files found in '{DATA_PATH}'")
    
    print(f"[LOAD DOCS] ✓ Successfully loaded {len(documents)} document(s)")
    for i, doc in enumerate(documents, 1):
        print(f"  - Document {i}: {doc.metadata.get('source', 'unknown')} ({len(doc.page_content)} chars)")
    
    return documents

def chunk_documents(documents):
    print("\n[CHUNK] Starting document chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    print(f"[CHUNK] ✓ Created {len(chunks)} chunks")
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    print(f"[CHUNK] Total characters in chunks: {total_chars}")
    
    return chunks

def create_vector_store(chunks):
    print("\n[EMBED] Creating embeddings and vector store...")
    print("[EMBED] Initializing OpenAI embeddings...")
    
    embeddings = OpenAIEmbeddings()
    
    print(f"[EMBED] Embedding {len(chunks)} chunks into vectors...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print("[EMBED] Persisting vector store to disk...")
    vector_store.persist()
    
    print(f"[EMBED] ✓ Vector store created and persisted to {PERSIST_DIRECTORY}")
    
    return vector_store

def initialize_rag_chain():
    global rag_chain
    
    print("\n" + "="*60)
    print("INITIALIZING RAG CHAIN")
    print("="*60)
    
    # Check if vector store exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"\n[INIT] Found existing Chroma database at {PERSIST_DIRECTORY}")
        print("[INIT] Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings()
        )
        print("[INIT] ✓ Vector store loaded")
    else:
        print(f"\n[INIT] No existing Chroma database found at {PERSIST_DIRECTORY}")
        print("[INIT] Creating new vector store from documents...")
        documents = load_documents()
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)
    
    # Initialize retriever
    print("\n[RETRIEVER] Setting up retriever with k=5...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("[RETRIEVER] ✓ Retriever initialized")
    
    # Test retriever
    print("\n[TEST] Testing retriever with sample query...")
    test_query = "developing countries health economic indicators"
    test_results = retriever.get_relevant_documents(test_query)
    print(f"[TEST] Query: '{test_query}'")
    print(f"[TEST] ✓ Retrieved {len(test_results)} chunks")
    if test_results:
        print(f"[TEST] First chunk preview: {test_results[0].page_content[:150]}...")
    
    # Initialize LLM
    print("\n[LLM] Initializing ChatOpenAI...")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
    print("[LLM] ✓ LLM initialized")
    
    # Create prompt
    print("\n[PROMPT] Creating prompt template...")
    prompt = ChatPromptTemplate.from_template(
        f"""{SYSTEM_PROMPT}

Context from documents:
{{context}}

Question: {{question}}"""
    )
    print("[PROMPT] ✓ Prompt template created")
    
    # Create RAG chain
    print("\n[CHAIN] Building RAG chain...")
    def format_docs(docs):
        formatted = "\n\n".join(doc.page_content for doc in docs)
        print(f"[CHAIN] Formatted {len(docs)} documents into context ({len(formatted)} chars)")
        return formatted
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("[CHAIN] ✓ RAG chain built successfully!")
    print("\n" + "="*60)
    print("RAG CHAIN INITIALIZATION COMPLETE")
    print("="*60 + "\n")

def chat(message, history):
    try:
        print(f"\n[CHAT] User query: '{message}'")
        response = rag_chain.invoke({"question": message})
        print(f"[CHAT] Response generated ({len(response)} chars)")
        return response
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[CHAT] ERROR: {error_msg}")
        return error_msg

iface = gr.ChatInterface(
    fn=chat,
    title="Economics Chatbot: Developing vs Developed Countries",
    description="Ask me anything about the economies of developing and developed countries.",
    theme=gr.themes.Soft(),
    examples=[
        "What are the main differences in GDP growth between developing and developed countries?",
        "How do renewable energy adoption rates differ between developed and developing nations?",
        "Why do developing countries have higher population growth rates?",
        "What role does FDI play in economic development?"
    ]
)

if __name__ == "__main__":
    initialize_rag_chain()
    iface.launch(share=False)