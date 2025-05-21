import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="RAG", layout="wide")

from RAG import initialize_index, search, llama2_ollama_generate

# Inject custom CSS for full-width buttons and centered sidebar title
st.markdown("""
    <style>
        div[data-testid="stSidebar"] {
            padding-top: 2rem;
        }
        .sidebar-title {
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        .sidebar-btn {
            width: 100%;
            display: block;
            background-color: #4CAF50;
            color: white;
            padding: 0.75rem 1rem;
            text-align: center;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
        }
        .sidebar-btn:hover {
            background-color: #388e3c;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation buttons using session state
st.sidebar.markdown('<div class="sidebar-title">📚 Menu</div>', unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Navigation buttons
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.page = "Home"

if st.sidebar.button("⚙️ Help", use_container_width=True):
    st.session_state.page = "Help"

# Cache model and index loading
@st.cache_resource
def get_index_and_data():
    return initialize_index(data_folder="samples")

# Load model and index once
model, index, all_sentences, metadata = get_index_and_data()

# ---------------------------------------
# PAGE: HOME
# ---------------------------------------
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center;'>RAG for Diagnostic Reasoning for Clinical Notes (DiReCT)</h1>", unsafe_allow_html=True)

    with st.form("query_form"):
        query = st.text_input("Ask a question based on data:")
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            submitted = st.form_submit_button("🔍 Search")

    if submitted and query.strip():
        with st.spinner("🔍 Retrieving relevant data and generating detailed answer..."):

            response = llama2_ollama_generate(query)

        st.subheader("💬 Answer")
        st.write(response)

# ---------------------------------------
# PAGE: HELP
# ---------------------------------------
elif st.session_state.page == "Help":
    st.markdown("<h1 style='text-align: center;'>⚙️ Help and Cache Management</h1>", unsafe_allow_html=True)
    st.write("Use this page to clear the cached model and index if you want to reload everything.")

    # Center the button using columns
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("🔄 Clear Cache"):
            st.cache_resource.clear()
            st.experimental_rerun()