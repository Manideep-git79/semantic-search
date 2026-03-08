from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from cache.semantic_cache import SemanticCache

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

cache = SemanticCache()

index = faiss.read_index("embeddings/faiss_index.index")

dataset = pd.read_csv("data/cleaned_dataset.csv")

class QueryRequest(BaseModel):
    query:str
    
@app.get("/")
def home():
    return {
        "message": "Semantic Search API is running",
        "docs": "Open /docs to test the API"
    }
    
@app.post("/query")
def query_search(request:QueryRequest):

    query=request.query

    hit,entry,score=cache.search(query)

    if hit:
        return {
            "query":query,
            "cache_hit":True,
            "matched_query":entry["query"],
            "similarity_score":float(score),
            "result":entry["result"],
            "dominant_cluster":None
        }

    query_embedding=model.encode([query])

    D,I=index.search(np.array(query_embedding),k=1)

    result=dataset.iloc[I[0][0]]["text"]

    cache.add(query,result)

    return {
        "query":query,
        "cache_hit":False,
        "matched_query":None,
        "similarity_score":None,
        "result":result,
        "dominant_cluster":None
    }


@app.get("/cache/stats")
def stats():
    return cache.stats()


@app.delete("/cache")
def clear():
    cache.clear()
    return {"message":"cache cleared"}
