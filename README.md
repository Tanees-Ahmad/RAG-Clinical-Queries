# 🔍💬 RAG-based QA System with Streamlit, FAISS, and Ollama

A lightweight Retrieval-Augmented Generation (RAG) application that lets you ask questions over custom JSON data using semantic search with **FAISS** and **MiniLM embeddings**, and generates answers via a **local LLM (Gemma 2B)** served with **Ollama**. The interface is powered by **Streamlit** for quick testing and demos.

---

## 🧠 Features

- 🗂️ **Custom JSON ingestion** and flattening for hierarchical keys
- 🔎 **Semantic search** using `sentence-transformers` and `FAISS`
- 🧬 Embedding with `all-MiniLM-L6-v2` (fast and efficient)
- 🧠 Query completion via `gemma:2b` LLM using Ollama
- 🧑‍💻 Simple and responsive **Streamlit interface**

---

## 🏗️ Architecture

      +-------------------+
      |  JSON Knowledge   |
      |     Base Files    |
      +--------+----------+
               |
               v
    +----------------------+
    |  Flatten & Embed     | ← all-MiniLM-L6-v2
    |  (SentenceTransform) |
    +----------------------+
               |
               v
    +----------------------+
    |     FAISS Index      |
    +----------+-----------+
               |
Query  →       v       →  Top-k Results
    +------------------------+
    |    Prompt Construction |
    +------------------------+
               |
               v
       +---------------+
       |  Ollama LLM   | ← gemma:2b
       +---------------+
               |
               v
         Final Answer



````markdown

---

### 1. Clone the Repository

```bash
git clone https://github.com/Tanees-Ahmad/RAG-Clinical-Queries.git
cd RAG-Clinical-Queries
````

---

### 2. Set Up Your Python Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install and Run Ollama with Gemma Model

1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Pull the Gemma model:

   ```bash
   ollama pull gemma:2b
   ```
3. Start the Ollama daemon (if not already running):

   ```bash
   ollama run gemma:2b
   ```

> ⚠️ Ensure that Ollama is accessible at `http://localhost:11434`.

---

### 5. Prepare Your JSON Knowledge Base

1. Create a folder named `samples/` in the project directory.
2. Place your `.json` files inside this folder.
3. Make sure your JSON files follow this format:

```json
{
  "input1": "Input text here",
  "output": {
    "topic1": {
      "key1": "value1",
      "key2": "value2"
    }
  },
  "input2": "Optional meta info"
}
```

---

### 6. Run the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

This will launch your web app in a browser where you can type questions and see AI-generated responses based on your custom JSON data.

---

### 7. Ask Questions in the Web UI

Example queries:

* "What are the key points under climate change?"
* "Summarize the recommendations section."

---

