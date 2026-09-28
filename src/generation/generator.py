from src.retrieval.reranker import hybrid_search_with_rerank
from src.generation.local_llm import local_chat

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.

Rules:
1. Answer only using information from the numbered context blocks below.
2. Cite your sources using bracketed numbers like [1], [2] right after the claim they support.
3. If the context does not contain enough information to answer, say so clearly instead of guessing.
4. Do not use any outside knowledge.
"""


def build_context_blocks(chunks):
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"]["source_file"]
        blocks.append(f"[{i}] (source: {source})\n{chunk['text']}")
    return "\n\n".join(blocks)


def generate_answer(query, strategy="structure_aware", top_k=5):
    chunks = hybrid_search_with_rerank(query, strategy, final_top_k=top_k)
    context = build_context_blocks(chunks)

    full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {query}

Answer the question using only the context above, with bracketed citations."""

    answer = local_chat(full_prompt)

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
    }


if __name__ == "__main__":
    query = "what should I do if a production service goes down"
    result = generate_answer(query)

    print("ANSWER:\n")
    print(result["answer"])
    print("\nSOURCES USED:\n")
    for i, chunk in enumerate(result["chunks"], start=1):
        print(f"[{i}] {chunk['metadata']['source_file']}")