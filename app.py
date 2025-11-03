import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Load FAISS and Corpus ---
st.title("🧠 MedBot - Conversational Medical Q&A")
st.write("Ask any medical question (from the MedQuAD dataset).")

@st.cache_resource
def load_data():
    index = faiss.read_index("medquad_index.faiss")
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, corpus, model

index, corpus, model = load_data()

# --- Query Input ---
query = st.text_input("🔍 Ask your question here:")

if query:
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), k=1)
    best_match = corpus[I[0][0]]
    st.subheader("💬 Answer:")
    st.write(best_match)
