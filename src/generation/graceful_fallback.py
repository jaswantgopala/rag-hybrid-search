from src.generation.generator import generate_answer
from src.generation.confidence import compute_confidence

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.5


def answer_with_fallback(query, strategy="structure_aware", top_k=5):
    result = generate_answer(query, strategy=strategy, top_k=top_k)
    scores = compute_confidence(query, result)

    if scores["retrieval_confidence"] < RETRIEVAL_CONFIDENCE_THRESHOLD:
        source_files = sorted(set(c["metadata"]["source_file"] for c in result["chunks"]))

        return {
            "status": "low_confidence",
            "answer": (
                "I couldn't find strong enough support in the indexed documents "
                "to answer this confidently."
            ),
            "closest_chunks_found": [
                {"source": c["metadata"]["source_file"], "preview": c["text"][:150]}
                for c in result["chunks"][:3]
            ],
            "documents_worth_checking_manually": source_files,
            "confidence_scores": scores,
        }

    return {
        "status": "answered",
        "answer": result["answer"],
        "sources": [c["metadata"]["source_file"] for c in result["chunks"]],
        "confidence_scores": scores,
    }


if __name__ == "__main__":
    good_query = "what should I do if a production service goes down"
    bad_query = "what is the company's parental leave policy"

    print("=== GOOD QUERY (should have real support) ===\n")
    result1 = answer_with_fallback(good_query)
    print(f"status: {result1['status']}")
    print(f"answer: {result1['answer']}\n")

    print("=== BAD QUERY (no support in corpus) ===\n")
    result2 = answer_with_fallback(bad_query)
    print(f"status: {result2['status']}")
    print(f"answer: {result2['answer']}")
    if result2["status"] == "low_confidence":
        print(f"documents worth checking: {result2['documents_worth_checking_manually']}")