from sentence_transformers import SentenceTransformer
import numpy as np
from data.load_data import load_dataset
from utils.preprocess import clean_text
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings():

    df = load_dataset()

    texts = [clean_text(t) for t in df["text"]]

    embeddings = model.encode(texts, show_progress_bar=True)

    np.save("embeddings/news_embeddings.npy", embeddings)

    df["clean_text"] = texts

    df.to_csv("data/cleaned_dataset.csv", index=False)

    print("Embeddings generated")

if __name__ == "__main__":
    generate_embeddings()