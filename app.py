import streamlit as st
from RAG import search, llama2_ollama_generate

st.set_page_config(page_title="RAG with Gemma", layout="wide")

st.title("📚 RAG Chat with Gemma")
st.write("Ask a question based on your custom JSON knowledge base.")

query = st.text_input("Enter your query:", placeholder="e.g., What is the process flow for XYZ?")
top_k = st.slider("Number of retrieved contexts:", min_value=1, max_value=10, value=5)

if st.button("Generate Answer") and query:
    with st.spinner("Retrieving relevant chunks..."):
        results = search(query, k=top_k)
        context = "\n".join([res[0] for res in results])

        full_prompt = f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        
    with st.spinner("Generating response using Gemma..."):
        response = llama2_ollama_generate(full_prompt)
        
    st.subheader("🔍 Retrieved Context")
    for i, (chunk, meta) in enumerate(results):
        st.markdown(f"**Chunk {i+1}:** {chunk}\n\n*Metadata:* `{meta}`")
    
    st.subheader("💬 Generated Answer")
    st.markdown(response)

# Optional styling
st.markdown(
    """
    <style>
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
