from rank_bm25 import BM25Okapi


def main():

    #prepare Documents
    documents = [
        "How to fix a car engine",
        "Automobile motor repair guide",
        "Best restaurants in Dubai Marina",
        "Vehicle maintenance and engine diagnostics"
    ]

    #our query
    query = "car engine repair"

    #BM25 Retrieval
    #Best Match algorithms for ranking used to search for the exact word

    # Tokenize
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())
    print("BM25 Scores:", bm25_scores)

    #Semantic Embeddings
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_embeddings = model.encode(documents)
    query_embedding = model.encode([query])[0]

    # Cosine similarity
    semantic_scores = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    print("Semantic Scores:", semantic_scores)

    # Hybrid Fusion
    # Normalize both scores
    #so score between 0 To 1
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)

    # Weighting factor
    alpha = 0.6
    # semantic weight
    hybrid_scores = alpha * semantic_norm + (1 - alpha) * bm25_norm
    print("Hybrid Scores:", hybrid_scores)

    # Ranking # -----------------------------
    #Sort the Ranking
    ranked_indices = hybrid_scores.argsort()[::-1]
    print("\nFinal Ranking (Hybrid):")
    for idx in ranked_indices:
        print(f"{documents[idx]} --> Score: {hybrid_scores[idx]:.4f}")

if __name__ == '__main__':
    main()

