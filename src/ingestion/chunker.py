import json
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str
    chunking_strategy: str
    char_count: int
    section_heading: str | None = None

def chunk_fixed_size(text: str, source_file: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Baseline: fixed-size windows with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]  # falls back to hard splits
    )
    pieces = splitter.split_text(text)
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            source_file=source_file,
            text=piece,
            chunking_strategy="fixed_size",
            char_count=len(piece)
        )
        for piece in pieces
    ]

def chunk_structure_aware(text: str, source_file: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Splits primarily on markdown/section headers, falling back to paragraphs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""]
    )
    pieces = splitter.split_text(text)
    chunks = []
    for piece in pieces:
        heading = None
        first_line = piece.strip().split("\n")[0]
        if first_line.startswith("#"):
            heading = first_line.lstrip("#").strip()
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            source_file=source_file,
            text=piece,
            chunking_strategy="structure_aware",
            char_count=len(piece),
            section_heading=heading
        ))
    return chunks

def chunk_semantic(text: str, source_file: str, similarity_threshold: float = 0.5) -> list[Chunk]:
    """Splits on topic boundaries: groups sentences until embedding similarity drops."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if len(sentences) <= 1:
        return [Chunk(
            chunk_id=str(uuid.uuid4()),
            source_file=source_file,
            text=text,
            chunking_strategy="semantic",
            char_count=len(text)
        )]

    embeddings = _embedder.encode(sentences)
    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        if sim < similarity_threshold:
            # topic boundary — close out current group
            group_text = ". ".join(current_group) + "."
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_file=source_file,
                text=group_text,
                chunking_strategy="semantic",
                char_count=len(group_text)
            ))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])

    if current_group:
        group_text = ". ".join(current_group) + "."
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            source_file=source_file,
            text=group_text,
            chunking_strategy="semantic",
            char_count=len(group_text)
        ))
    return chunks

STRATEGIES = {
    "fixed_size": chunk_fixed_size,
    "structure_aware": chunk_structure_aware,
    "semantic": chunk_semantic,
}

def chunk_all_documents(
    processed_path: str = "data/processed/processed_docs.jsonl",
    output_dir: str = "data/processed",
    strategy: str = "all"
) -> dict[str, list[Chunk]]:
    """Run one or all chunking strategies over every processed doc."""
    docs = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    strategies_to_run = STRATEGIES.keys() if strategy == "all" else [strategy]
    results: dict[str, list[Chunk]] = {}

    for strat_name in strategies_to_run:
        strat_fn = STRATEGIES[strat_name]
        all_chunks = []
        for doc in docs:
            if strat_name == "semantic":
                chunks = strat_fn(doc["text"], doc["source_file"])
            else:
                chunks = strat_fn(doc["text"], doc["source_file"])
            all_chunks.extend(chunks)
        results[strat_name] = all_chunks

        out_file = Path(output_dir) / f"chunks_{strat_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        print(f"[{strat_name}] {len(all_chunks)} chunks -> {out_file}")

    return results

if __name__ == "__main__":
    chunk_all_documents()