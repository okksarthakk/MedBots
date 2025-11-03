import streamlit as st
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load FAISS index and corpus
# -------------------------------
@st.cache_resource
def load_resources():
    index = faiss.read_index("medquad_index.faiss")
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, corpus, model

index, corpus, model = load_resources()

# -------------------------------
# Helper function
# -------------------------------
def get_answer(query):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype('float32'), k=1)
    best_match = corpus[I[0][0]]
    return best_match

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🧠 MedBot", page_icon="💬", layout="wide")

# Sidebar section
with st.sidebar:
    st.title("💬 MedBot - Medical Q&A")
    st.markdown("""
    **Developed by:** Sarthak Sarode  
    **Dataset:** [MedQuAD](https://github.com/abachaa/MedQuAD)  
    **Model:** all-MiniLM-L6-v2  
    **Description:**  
    This chatbot retrieves medically accurate answers from the MedQuAD dataset using FAISS for semantic similarity search.
    """)
    st.info("⚠️ Disclaimer: This chatbot is for educational purposes only and not a substitute for professional medical advice.")

st.title("🧠 MedBot - Conversational Medical Q&A")

# Initialize session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
user_query = st.chat_input("Ask a medical question...")

if user_query:
    with st.spinner("Searching for answer..."):
        response = get_answer(user_query)
        st.session_state.chat_history.append({"query": user_query, "answer": response})

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
