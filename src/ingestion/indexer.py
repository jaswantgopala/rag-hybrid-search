import json
import pickle
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

embedder = SentenceTransformer("all-MiniLM-L6-v2")
DUPLICATE_THRESHOLD = 0.95


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def tokenize(text):
    return text.lower().split()


def build_index(strategy, chunks_path_template="data/processed/chunks_{strategy}.jsonl",
                 chroma_dir="data/chroma_db", bm25_dir="data/bm25_index"):

    chunks_path = chunks_path_template.format(strategy=strategy)
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"[{strategy}] loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    kept_chunks = []
    kept_embeddings = []
    skipped = 0

    for chunk, emb in zip(chunks, embeddings):
        dup = False
        for kept_emb in kept_embeddings:
            if cosine_sim(emb, kept_emb) > DUPLICATE_THRESHOLD:
                dup = True
                break
        if dup:
            skipped += 1
            continue
        kept_chunks.append(chunk)
        kept_embeddings.append(emb)

    print(f"[{strategy}] kept {len(kept_chunks)}, skipped {skipped} duplicates")

    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection_name = f"docs_{strategy}"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    collection.add(
        ids=[c["chunk_id"] for c in kept_chunks],
        embeddings=[emb.tolist() for emb in kept_embeddings],
        documents=[c["text"] for c in kept_chunks],
        metadatas=[
            {
                "source_file": c["source_file"],
                "chunking_strategy": c["chunking_strategy"],
                "char_count": c["char_count"],
                "section_heading": c.get("section_heading") or "",
            }
            for c in kept_chunks
        ],
    )
    print(f"[{strategy}] stored {len(kept_chunks)} chunks in collection '{collection_name}'")

    Path(bm25_dir).mkdir(parents=True, exist_ok=True)
    tokenized_corpus = [tokenize(c["text"]) for c in kept_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in kept_chunks],
        "texts": [c["text"] for c in kept_chunks],
        "metadatas": [
            {"source_file": c["source_file"], "chunking_strategy": c["chunking_strategy"]}
            for c in kept_chunks
        ],
    }

    bm25_path = Path(bm25_dir) / f"bm25_{strategy}.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"[{strategy}] bm25 index saved to {bm25_path}")
    return collection, bm25_data


if __name__ == "__main__":
    for strategy in ["fixed_size", "structure_aware", "semantic"]:
        build_index(strategy)
        print()