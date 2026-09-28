from src.generation.citation_verifier import extract_claims_with_citations, verify_answer
from src.generation.local_llm import local_chat_json


def retrieval_confidence(chunks):
    if not chunks:
        return 0.0
    top_score = chunks[0].get("rerank_score", 0)
    normalized = 1 / (1 + pow(2.718281828, -top_score))
    return round(normalized, 3)


def citation_coverage(answer_text, verification_report):
    total_sentences = len([s for s in answer_text.split(".") if s.strip()])
    cited_sentences = len(extract_claims_with_citations(answer_text))

    if not verification_report:
        coverage_ratio = 0.0
    else:
        verified_count = sum(1 for entry in verification_report if entry["supported"])
        coverage_ratio = verified_count / len(verification_report)

    sentence_coverage = cited_sentences / total_sentences if total_sentences else 0.0

    return {
        "verified_citation_ratio": round(coverage_ratio, 3),
        "sentence_citation_ratio": round(sentence_coverage, 3)
    }


def answer_completeness(query, answer_text):
    prompt = f"""Question: {query}

Answer: {answer_text}

Does this answer fully address all parts of the question? Respond with only valid JSON:
{{"completeness_score": a number from 0.0 to 1.0, "reason": "one short sentence"}}"""

    return local_chat_json(prompt, fallback={"completeness_score": 0.5, "reason": "could not parse"})


def compute_confidence(query, result):
    answer_text = result["answer"]
    chunks = result["chunks"]

    verification_report = verify_answer(result)

    retr_conf = retrieval_confidence(chunks)
    cit_cov = citation_coverage(answer_text, verification_report)
    completeness = answer_completeness(query, answer_text)

    composite = round(
        (retr_conf * 0.4) +
        (cit_cov["verified_citation_ratio"] * 0.35) +
        (completeness["completeness_score"] * 0.25),
        3
    )

    return {
        "retrieval_confidence": retr_conf,
        "citation_coverage": cit_cov,
        "completeness": completeness,
        "composite_score": composite,
        "verification_report": verification_report
    }


if __name__ == "__main__":
    from src.generation.generator import generate_answer

    query = "what should I do if a production service goes down"
    result = generate_answer(query)

    print("ANSWER:\n")
    print(result["answer"])

    scores = compute_confidence(query, result)

    print("\nCONFIDENCE BREAKDOWN:\n")
    print(f"Retrieval confidence: {scores['retrieval_confidence']}")
    print(f"Citation coverage: {scores['citation_coverage']}")
    print(f"Completeness: {scores['completeness']}")
    print(f"\nCOMPOSITE SCORE: {scores['composite_score']}")