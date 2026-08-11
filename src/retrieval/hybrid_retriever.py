import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.database.vector_store import load_existing_vector_store
from src.ingestion.document_loader import load_pdf, chunk_documents

def build_hybrid_retriever(chunks, vector_store):
    """
    Combines BM25 (keyword search) and Vector Database (semantic search)
    using Reciprocal Rank Fusion (RRF).
    """
    print("Initializing BM25 Keyword Retriever...")
    
    # 1. Initialize the BM25 Retriever with our chunks
    bm25_retriever = BM25Retriever.from_documents(chunks)
    # We ask BM25 for the top 5 exact keyword matches
    bm25_retriever.k = 5 
    
    # 2. Initialize the Vector Retriever
    # We ask Chroma for the top 5 semantic matches
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print("Combining into Ensemble Retriever (Hybrid Search)...")
    
    # 3. Combine them using LangChain's EnsembleRetriever
    # weights=[0.4, 0.6] means we give slightly more importance to semantic meaning (60%) 
    # than pure keyword matching (40%), which is standard for RAG.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    return hybrid_retriever

# --- Local Testing Block ---
if __name__ == "__main__":
    SAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/sample_climate_report.pdf")
    
    try:
        print("Loading chunks for BM25...")
        pages = load_pdf(SAMPLE_PDF_PATH)
        doc_chunks = chunk_documents(pages)
        
        print("Loading Chroma Database...")
        db = load_existing_vector_store()
        
        # Build the Hybrid Retriever
        retriever = build_hybrid_retriever(doc_chunks, db)
        
        test_query = "What is the exact target for global surface temperature?"
        print(f"\n--- Testing Hybrid Search ---")
        print(f"Query: '{test_query}'\n")
        
        # Invoke the retriever directly (bypassing the LLM for now to see raw chunks)
        results = retriever.invoke(test_query)
        
        print(f"Retrieved {len(results)} unique chunks (merged from Vector and BM25):")
        for i, res in enumerate(results[:3]): # Just print top 3 to keep terminal clean
            print(f"\nRank {i+1} (Page {res.metadata.get('page', 'Unknown')}):")
            print(f"{res.page_content[:200]}...")
            
    except Exception as e:
        print(f"Failed during hybrid retrieval: {e}")