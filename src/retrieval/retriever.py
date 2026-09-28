import pickle
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize(text):
    return text.lower().split()


def dense_retrieve(query, strategy, chroma_dir="data/chroma_db", k=10):
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(f"docs_{strategy}")

    query_embedding = embedder.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i]
        })

    return hits


def sparse_retrieve(query, strategy, bm25_dir="data/bm25_index", k=10):
    bm25_path = Path(bm25_dir) / f"bm25_{strategy}.pkl"
    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)

    tokenized_query = tokenize(query)
    scores = bm25_data["bm25"].get_scores(tokenized_query)

    scored = list(zip(scores, range(len(scores))))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    hits = []
    for score, idx in top:
        hits.append({
            "chunk_id": bm25_data["chunk_ids"][idx],
            "text": bm25_data["texts"][idx],
            "metadata": bm25_data["metadatas"][idx],
            "score": float(score)
        })

    return hits


if __name__ == "__main__":
    query = "how do I reset my password"
    strategy = "structure_aware"

    print("dense results:")
    for hit in dense_retrieve(query, strategy, k=5):
        print(f"  {hit['score']:.3f} | {hit['text'][:80]}")

    print("\nsparse results:")
    for hit in sparse_retrieve(query, strategy, k=5):
        print(f"  {hit['score']:.3f} | {hit['text'][:80]}")