# Wikipedia RAG

strong retreival system + simple generative model = smart context-aware result ❤️

## Summary

simple RAG system that takes wikipedia articles based on your topic, processes them, and lets you ask questions in plain language.
by integrating the retreival system with a generative model to find the most relevant information and then explains it to you.

## Objective

build a simple RAG system using `faiss`, `sentence-transformers` and `transformers`, and combine them togther using `langchain`. deploy the application using `fastapi` and `docker`.

## Methodology
1. for the documents:
```
topic -> load the docuemnts -> chunk the documents -> embedding the chunks -> store the embeddings.
```
2. for the query:
```
query -> embedding the query -> similarity search -> retrieve the top-k similar chunks. 
```
3. chaining:

```
combine the retrieved chunks as a context -> combine the context with the query to have a single prompt using chat template -> send the promt to the llm -> context-aware result. 
```
## Usage

clone the repo, install the requirements.txt, run the main.py.
