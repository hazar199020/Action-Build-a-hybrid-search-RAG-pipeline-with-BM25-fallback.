# 🔍 Hybrid Search Engine (BM25 + Semantic Search)

A simple and effective hybrid search pipeline combining **BM25** (lexical search) with **semantic embeddings** to achieve high‑quality retrieval for RAG and information‑retrieval tasks.

This project demonstrates how to:
- Build a BM25 retriever  
- Build a semantic embedding retriever  
- Normalize and combine scores  
- Produce hybrid ranked results  
- Compare performance between methods  

---

## 🚀 Features

- **BM25 lexical search** using `rank_bm25`  
- **Semantic search** using sentence embeddings  
- **Hybrid scoring** (weighted combination)  
- **Score normalization**  
- **Easy‑to‑extend pipeline**  
- **Clean, modular Python code**  

---

## 📌 Why Hybrid Search?

BM25 is great for:
- exact keyword matching  
- short queries  
- sparse text  

Semantic search is great for:
- meaning‑based retrieval  
- synonyms  
- long queries  

Hybrid search gives you the **best of both worlds**.

---

## 🧠 Architecture Overview

