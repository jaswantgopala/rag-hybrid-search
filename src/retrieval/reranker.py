from sentence_transformers import CrossEncoder
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.retriever import dense_retrieve, sparse_retrieve

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, candidates, top_k=5):
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


def hybrid_search_with_rerank(query, strategy, dense_k=10, sparse_k=10, fusion_top_n=20, final_top_k=5):
    dense_hits = dense_retrieve(query, strategy, k=dense_k)
    sparse_hits = sparse_retrieve(query, strategy, k=sparse_k)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
    candidates = fused[:fusion_top_n]
    return rerank(query, candidates, top_k=final_top_k)


if __name__ == "__main__":
    query = "what happens if a production service goes down"
    strategy = "structure_aware"

    results = hybrid_search_with_rerank(query, strategy)

    for hit in results:
        print(f"{hit['rerank_score']:.4f} | {hit['metadata']['source_file']} | {hit['text'][:80]}")