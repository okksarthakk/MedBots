import os
os.system("pip install faiss-cpu")

import faiss

import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS Index and Corpus

@st.cache_resource
def load_data():
    index = faiss.read_index("medquad_index.faiss")
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, corpus, model

index, corpus, model = load_data()

# Helper function
def get_answer(query):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), k=1)
    best_match = corpus[I[0][0]]
    return best_match

# Sidebar Section

with st.sidebar:
    st.title("🧠 MedBot - Medical Q&A")
    st.markdown("""
    **Developed by:** Sarthak Sarode  
    **Dataset:** [MedQuAD](https://github.com/abachaa/MedQuAD)  
    **Model:** all-MiniLM-L6-v2  
    **Description:**  
    This chatbot retrieves medically accurate answers using FAISS for semantic similarity search.
    """)
    st.info("⚠️ This chatbot is for educational purposes only and **not** a substitute for professional medical advice.")

# Main Chat UI

st.title("💬 MedBot - Conversational Medical Q&A")
st.write("Ask any medical question (from the MedQuAD dataset).")

# Initialize session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_query = st.chat_input("Ask a medical question...")

if user_query:
    with st.spinner("Searching for an answer..."):
        response = get_answer(user_query)
        st.session_state.chat_history.append({"query": user_query, "answer": response})

# Display chat messages
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['query']}")
    with st.chat_message("assistant"):
        st.markdown(f"**MedBot:** {chat['answer']}")

# Footer style
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("💡 *MedBot uses semantic search powered by FAISS & Sentence Transformers.*")
