# Semantic Search API

This project implements a semantic search system using the 20 Newsgroups dataset.

## Features
- Semantic search
- Document clustering
- Caching for faster queries
- FastAPI based API

## Architecture

User Query → Embedding → Semantic Cache → Vector Search → Result

If a similar query already exists in cache, the cached result is returned without recomputation.

## API Endpoints

POST /query

Request:
{
  "query": "machine learning algorithms"
}

Response:
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}

GET /cache/stats

Returns cache statistics.

DELETE /cache

Clears the cache.

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run server:

uvicorn main:app --reload

Open API docs:

http://127.0.0.1:8000/docs
