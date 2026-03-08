import faiss
import numpy as np

def create_index():

    embeddings = np.load("embeddings/news_embeddings.npy")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index,"embeddings/faiss_index.index")

    print("FAISS index created")

if __name__ == "__main__":
    create_index()