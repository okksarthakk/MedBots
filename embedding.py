# embeddings.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\sarth\OneDrive\Desktop\Coding\nlp\medquad.csv")

print("CSV Loaded Successfully")
print(data.head())

# Quick check of columns
print("Columns in dataset:", data.columns)
print("Number of rows:", len(data))

# Combine question and answer into one text block
# (This makes retrieval more contextual)
data["text"] = data["question"] + " " + data["answer"]

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loapython embeddings.pyded successfully ")

# Encode each text into a dense vector
corpus = data["text"].tolist()
embeddings = model.encode(corpus, show_progress_bar=True)

# Store embeddings in FAISS index
# Normalize embeddings for cosine similarity
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = inner product = cosine similarity
index.add(embeddings)

print(f"Added {len(corpus)} vectors to FAISS index ")

# Save for later use
faiss.write_index(index, "medquad_index.faiss")

# Save model and dataframe for retrieval
with open("corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)

print("Embeddings and FAISS index saved successfully ")
