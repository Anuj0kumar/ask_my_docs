import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import our database loader from the previous module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.database.vector_store import load_existing_vector_store

def get_llm():
    """
    Initializes the local LLM. 
    We use temperature=0 to make the model deterministic (factual, not creative).
    """
    return ChatOllama(model="llama3.2", temperature=0)

def build_rag_chain(vector_store):
    """
    Builds the retrieval-augmented generation chain.
    It connects the vector database search to the LLM via a strict prompt.

    """
    llm = get_llm()
    
    # 1. Define the System Prompt
    system_prompt = (
        "You are a highly precise environmental policy assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not contained within the context, say 'I cannot answer this based on the provided documents.' "
        "Do not use outside knowledge. "
        "\n\n"
        "Context:\n{context}"
    )

    # 2. Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 3. Create the document chain (injects the chunks into the {context} variable)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Turn the vector store into a retriever
    # search_kwargs={"k": 3} means we only grab the top 3 chunks to save context window space
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 5. Combine them into the final RAG chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


    


# --- Local Testing Block ---
if __name__ == "__main__":
    try:
        print("Loading Chroma Database...")
        db = load_existing_vector_store()
        
        print("Building RAG Chain...")
        chain = build_rag_chain(db)
        
        test_query = "What are the impacts of greenhouse gas emissions?"
        print(f"\nUser Query: {test_query}")
        print("Thinking...\n")
        
        # Execute the chain
        response = chain.invoke({"input": test_query})

        """    
            When you pass an {"input": query} to it, it automatically takes the query, 
            searches the vector store, grabs the top 3 chunks, 
            stuffs them into the prompt, and sends the whole package to the LLM.
            LLM will generate responce based on that.
            
        """
        
        print("--- FINAL ANSWER ---")
        print(response["answer"])
        print("--------------------")
        
        print("\n--- CITED SOURCES ---")
        for doc in response["context"]:
            print(f"- Page {doc.metadata.get('page', 'Unknown')}")
            
    except Exception as e:
        print(f"Failed during RAG execution: {e}")


    