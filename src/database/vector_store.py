import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Define where we want to save our database on the hard drive
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/chroma_db")

def get_embedding_model():
    """
    Initializes the local embedding model via Ollama.
    nomic-embed-text is highly optimized for RAG retrieval tasks.
    """
    return OllamaEmbeddings(model="nomic-embed-text")

def build_vector_store(chunks):
    """
    Takes text chunks, embeds them, and saves them to a local Chroma database.
    """
    print(f"Initializing embedding model and building Chroma DB at {DB_DIR}...")
    print("This may take a minute or two depending on your CPU/GPU...")
    
    embedding_model = get_embedding_model()
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    
    print("Vector database successfully built and saved to disk!")
    return vector_store

def load_existing_vector_store():
    """
    Loads a previously built Chroma database from disk.
    """
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError("Chroma DB not found. Please build it first.")
        
    embedding_model = get_embedding_model()
    vector_store = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embedding_model
    )
    return vector_store

# --- Local Testing Block ---
if __name__ == "__main__":
    # We need to import our loader from the other module to test the full flow
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.ingestion.document_loader import load_pdf, chunk_documents
    
    SAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/sample_climate_report.pdf")
    
    try:
        # 1. Load and chunk (Reusing our Milestone 1 code!)
        pages = load_pdf(SAMPLE_PDF_PATH)
        doc_chunks = chunk_documents(pages)
        
        # 2. Build the vector store
        db = build_vector_store(doc_chunks)
        
        # 3. Test a vector search!
        test_query = "What are the impacts of greenhouse gas emissions?"
        print(f"\n--- Testing Semantic Search ---")
        print(f"Query: '{test_query}'")
        
        # Fetch the top 2 most mathematically similar chunks
        results = db.similarity_search(test_query, k=2)
        
        for i, res in enumerate(results):
            print(f"\nResult {i+1} (Page {res.metadata.get('page', 'Unknown')}):")
            print(f"{res.page_content[:200]}...")
            
    except Exception as e:
        print(f"Failed during database build: {e}")