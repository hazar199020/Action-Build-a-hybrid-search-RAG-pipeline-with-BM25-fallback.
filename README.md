# 🔍 Hybrid Search Engine (BM25 + Semantic Search + Reranker)

This project demonstrates a complete retrieval pipeline combining:

- **BM25 lexical search**
- **Semantic search using sentence embeddings**
- **Hybrid weighted fusion**
- **Cross‑encoder reranking for final precision**

This mirrors how modern RAG systems retrieve and refine documents before passing them to an LLM.

---

## 🚀 Features

- BM25 retrieval using `rank_bm25`
- Semantic similarity using `sentence-transformers`
- Hybrid scoring (normalized + weighted)
- Cross‑encoder reranker for final ranking
- Simple, readable Python implementation

---

## 📌 Why Hybrid + Reranker?

Each retrieval method has strengths:

- **BM25** → great for exact keyword matching  
- **Semantic embeddings** → great for meaning and synonyms  
- **Hybrid fusion** → balances both  
- **Reranker** → reads *query + document together* and produces the most accurate final ranking

  ---

  ## 📊 Performance Benchmarks

| Method              | Recall@4 | Latency (ms) |
|---------------------|----------|------------- |
| BM25                | 0.9311   | 0.60         |
| Semantic Search     | 0.7563   | 21318.5      |
| Hybrid (BM25+SEM)   | 0.9364   | 0.36         |
| Reranker (Final)    | 3.1045   | 13124        |


This pipeline improves both **recall** and **precision**, which is essential for RAG and QA systems.

---
