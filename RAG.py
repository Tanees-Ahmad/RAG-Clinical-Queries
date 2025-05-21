# rag.py
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

def load_json_data(root_folder):
    data_list = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(subdir, file), 'r') as f:
                    try:
                        data = json.load(f)
                        data_list.append(data)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
    return data_list

def flatten_json_tree(tree, parent_key=""):
    flattened = []
    for key, value in tree.items():
        if key.startswith("input"):
            continue
        full_key = f"{parent_key} → {key}" if parent_key else key
        if isinstance(value, dict) and value:
            flattened.extend(flatten_json_tree(value, full_key))
        else:
            flattened.append(full_key)
    return flattened

model = SentenceTransformer('all-MiniLM-L6-v2')

data_list = load_json_data("samples")
all_sentences = []
metadata = []

for data in data_list:
    flat_sentences = flatten_json_tree(data)
    all_sentences.extend(flat_sentences)
    metadata.extend([data.get("input2", "")] * len(flat_sentences))

embeddings = model.encode(all_sentences, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def search(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    return [(all_sentences[i], metadata[i]) for i in I[0]]

def llama2_ollama_generate(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma:2b", "prompt": prompt},
        stream=True
    )

    result = "" 
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if 'response' in data:
                result += data['response']
    
    return result
