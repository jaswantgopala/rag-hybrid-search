from src.retrieval.retriever import dense_retrieve, sparse_retrieve


def reciprocal_rank_fusion(dense_hits, sparse_hits, dense_weight=0.7, sparse_weight=0.3, k=60):
    scores = {}
    chunk_data = {}

    for rank, hit in enumerate(dense_hits):
        chunk_id = hit["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + dense_weight * (1 / (k + rank + 1))
        chunk_data[chunk_id] = hit

    for rank, hit in enumerate(sparse_hits):
        chunk_id = hit["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + sparse_weight * (1 / (k + rank + 1))
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = hit

    fused = []
    for chunk_id, score in scores.items():
        entry = dict(chunk_data[chunk_id])
        entry["fused_score"] = score
        fused.append(entry)

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused


def hybrid_search(query, strategy, dense_k=10, sparse_k=10, top_n=5, dense_weight=0.7, sparse_weight=0.3):
    dense_hits = dense_retrieve(query, strategy, k=dense_k)
    sparse_hits = sparse_retrieve(query, strategy, k=sparse_k)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, dense_weight, sparse_weight)
    return fused[:top_n]


if __name__ == "__main__":
    query = "what happens if a production service goes down"
    strategy = "structure_aware"

    results = hybrid_search(query, strategy, top_n=5)

    for hit in results:
        print(f"{hit['fused_score']:.4f} | {hit['metadata']['source_file']} | {hit['text'][:80]}")