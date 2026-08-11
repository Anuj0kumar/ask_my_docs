import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str):
    """
    Loads a PDF file and extracts its pages using PyMuPDF.
    PyMuPDF is highly accurate for complex layouts (like IPCC reports)
    and automatically attaches the page number to the metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
    
    print(f"Loading PDF from: {file_path}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    print(f"Successfully loaded {len(documents)} pages.")
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """
    Splits a list of documents into smaller, overlapping chunks.
    RecursiveCharacterTextSplitter tries to split on paragraphs first, 
    then sentences, then words, keeping related ideas together.
    """
    print(f"Chunking documents (Size: {chunk_size}, Overlap: {chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""] # Tries to split by double newline (paragraph) first
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} total chunks.")
    return chunks

# --- Local Testing Block ---
# This block only runs if we execute this specific file directly.
if __name__ == "__main__":
    # 1. Provide a path to a test PDF in your data folder
    # IMPORTANT: You must put a sample PDF in this folder first!
    SAMPLE_PDF_PATH = "../../data/raw/sample_climate_report.pdf" 
    
    # Resolve the absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_pdf_path = os.path.normpath(os.path.join(script_dir, SAMPLE_PDF_PATH))
    
    try:
        # Load the document
        pages = load_pdf(absolute_pdf_path)
        
        # Chunk the document
        doc_chunks = chunk_documents(pages)
        
        # Print a sample to verify it worked
        if doc_chunks:
            print("\n--- SAMPLE CHUNK (First Chunk) ---")
            print(f"Content: {doc_chunks[0].page_content[:200]}...") 
            print(f"Metadata (Source & Page): {doc_chunks[0].metadata}")
            print("----------------------------------")
            
    except Exception as e:
        print(f"Failed during testing: {e}")