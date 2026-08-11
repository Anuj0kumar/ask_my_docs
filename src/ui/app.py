import streamlit as st
import os
import sys

# Ensure Python can find our src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.vector_store import load_existing_vector_store
from src.retrieval.hybrid_retriever import build_hybrid_retriever
from src.retrieval.reranker import build_advanced_retriever
from src.retrieval.naive_rag import build_rag_chain

# --- 1. UI Configuration ---
st.set_page_config(page_title="Ask My Docs", page_icon="🌍", layout="centered")
st.title("🌍 Ask My Docs: Climate Policy")
st.markdown("Ask questions about the IPCC Synthesis Report. Answers are strictly cited.")

# --- 2. Application State Management ---
# @st.cache_resource ensures we only load the heavy AI models and DB once, not on every click.
@st.cache_resource
def initialize_backend():
    print("Initializing backend services...")
    db = load_existing_vector_store()
    
    # For a production app, we would load the actual documents here, 
    # but to save time for this UI test, we will fetch chunks directly from the DB's internal memory
    db_data = db.get()
    
    # Reconstruct LangChain Document objects for the BM25 retriever
    from langchain_core.documents import Document
    chunks = [Document(page_content=txt, metadata=meta) for txt, meta in zip(db_data['documents'], db_data['metadatas'])]
    
    hybrid_retriever = build_hybrid_retriever(chunks, db)
    advanced_retriever = build_advanced_retriever(hybrid_retriever, top_k=3)
    
    # We modify our original rag_chain builder to accept the advanced retriever
    from src.retrieval.naive_rag import get_llm
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = get_llm()
    system_prompt = (
        "You are a precise environmental policy assistant. "
        "Use ONLY the retrieved context to answer the question. "
        "If the answer is not in the context, say 'I cannot answer this based on the provided documents.'\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    
    # Wire the advanced retriever into the final RAG chain
    final_chain = create_retrieval_chain(advanced_retriever, qa_chain)
    return final_chain

# Load the backend into Streamlit's memory
try:
    rag_chain = initialize_backend()
except Exception as e:
    st.error(f"Backend failed to load. Did you build the Chroma DB first? Error: {e}")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. UI Render Loop ---
# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_query := st.chat_input("Ask a question about the climate report..."):
    # Add user message to UI and state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and analyzing..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            # Format the citations
            citations = "\n\n**Sources:**\n"
            for doc in response["context"]:
                page = doc.metadata.get('page', 'Unknown')
                citations += f"- Page {page}\n"
            
            full_response = answer + citations
            st.markdown(full_response)
            
    # Save AI response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})