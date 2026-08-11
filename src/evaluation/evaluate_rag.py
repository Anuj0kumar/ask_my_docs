import os
import sys
import json
import pandas as pd
from datasets import Dataset

# --- 1. MONKEY PATCH FOR RAGAS BUG (MUST BE ABSOLUTELY FIRST) ---
# This prevents the VertexAI crash before Ragas even loads.
import types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module
# -----------------------------------------------------------------

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.run_config import RunConfig  # <-- This is the key to preventing timeouts

# Import Local Wrappers for our Judge
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import our backend pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.ui.app import initialize_backend

def load_golden_dataset(filepath: str):
    """Loads the manually verified Q&A pairs."""
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation():
    print("1. Initializing Local Backend Pipeline...")
    rag_chain = initialize_backend()
    
    dataset_path = os.path.join(os.path.dirname(__file__), "../../data/evaluation/golden_dataset.json")
    golden_data = load_golden_dataset(dataset_path)
    
    print(f"2. Processing {len(golden_data)} test cases via Local Ollama...")
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for item in golden_data:
        question = item["question"]
        print(f"   Asking: '{question}'")
        
        response = rag_chain.invoke({"input": question})
        retrieved_contexts = [doc.page_content for doc in response["context"]]
        
        data_samples["question"].append(question)
        data_samples["answer"].append(response["answer"])
        data_samples["contexts"].append(retrieved_contexts)
        data_samples["ground_truth"].append(item["ground_truth"])
        
    eval_dataset = Dataset.from_dict(data_samples)
    
    print("\n3. Initializing Ragas Judge (Local Ollama)...")
    # We use local Ollama to judge the answers. No API keys required!
    judge_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    
    print("4. Running Ragas Evaluation (Sequential processing, this will take a few minutes)...")
    
    # --- The Hardware Fix ---
    # max_workers=1 forces Ragas to process one question at a time so Ollama doesn't crash.
    # timeout=600 gives Ollama a full 10 minutes per question if needed.
    local_config = RunConfig(timeout=600, max_workers=1)
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=local_config
    )
    
    print("\n--- EVALUATION RESULTS ---")
    df = results.to_pandas()
    print(df[['question', 'faithfulness', 'context_precision']])
    print("\n--- AGGREGATE SCORES ---")
    print(results)

if __name__ == "__main__":
    run_evaluation()