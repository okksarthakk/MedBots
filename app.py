import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------------------------
# Load Data and Model
# -------------------------
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv('medquad.csv')
    index = faiss.read_index('medquad_index.faiss')
    return model, df, index

model, df, index = load_model_and_data()

# -------------------------
# Streamlit App Layout
# -------------------------
st.set_page_config(page_title="🩺 MedBot - Medical Q&A Assistant", page_icon="💬", layout="centered")

st.title("🩺 MedBot")
st.markdown("### Your AI Medical Assistant")
st.write("Ask any medical question based on the **MedQuAD dataset** below:")

# -------------------------
# User Query Input
# -------------------------
query = st.text_input("💬 Enter your medical question:")

if query:
    # Encode query and search FAISS index
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k=3)

    # Retrieve top results
    results = df.iloc[indices[0]]
    
    st.markdown("### 🧠 Top Answers")
    for i, row in results.iterrows():
        st.markdown(f"**Q:** {row['question']}")
        st.markdown(f"**A:** {row['answer']}")
        st.write("---")
else:
    st.info("Type a question to get started...")

st.markdown("⚕️ *Powered by FAISS and Sentence Transformers*")
