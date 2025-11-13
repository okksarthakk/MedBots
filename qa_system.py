import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load saved data
print("Loading FAISS index and corpus...")
index = faiss.read_index("medquad_index.faiss")

with open("corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Models loaded successfully ✅\n")

print("🤖 Medical QA Chatbot Ready!\n")

def get_answer(query):
    # Encode query
    query_vector = model.encode([query])
    
    # Search in FAISS index
    D, I = index.search(np.array(query_vector).astype('float32'), k=1)
    
    # Retrieve most similar text
    best_match = corpus[I[0][0]]
    distance = D[0][0]

    print(f"\n🩺 Question: {query}")
    print(f"🔍 FAISS distance: {distance:.4f}")

    # Lower distance = more similarity
    if distance < 0.6:
        print("❌ Sorry, I couldn't find an answer.")
    else:
        print(f"💬 Closest Match: {best_match}")

while True:
    user_query = input("\nAsk a medical question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break
    get_answer(user_query)
