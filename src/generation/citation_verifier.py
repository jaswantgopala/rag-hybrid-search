import re
from src.generation.generator import generate_answer
from src.generation.local_llm import local_chat_json

VERIFY_PROMPT = """You are a strict fact-checker. You will be given a claim and a source passage.
Decide if the source passage actually supports the claim.

Respond with only valid JSON in this exact format, nothing else:
{{"supported": true or false, "reason": "one short sentence"}}

Claim: {claim}

Source passage: {source}
"""


def extract_claims_with_citations(answer_text):
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    claims = []
    last_sentence_text = None
    last_claim_index = None

    for sentence in sentences:
        citations = [int(c) for c in re.findall(r'\[(\d+)\]', sentence)]
        clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()

        if clean_sentence:
            last_sentence_text = clean_sentence

        if not citations:
            continue

        if clean_sentence:
            claims.append({
                "claim": clean_sentence,
                "citation_numbers": citations
            })
            last_claim_index = len(claims) - 1
        elif last_claim_index is not None:
            claims[last_claim_index]["citation_numbers"].extend(citations)
        elif last_sentence_text:
            claims.append({
                "claim": last_sentence_text,
                "citation_numbers": citations
            })
            last_claim_index = len(claims) - 1

    return claims


def verify_claim(claim_text, source_text):
    prompt = VERIFY_PROMPT.format(claim=claim_text, source=source_text)
    return local_chat_json(prompt, fallback={"supported": None, "reason": "could not parse verifier response"})


def verify_answer(result):
    answer_text = result["answer"]
    chunks = result["chunks"]

    claims = extract_claims_with_citations(answer_text)
    verification_report = []

    for claim in claims:
        for citation_num in claim["citation_numbers"]:
            if citation_num < 1 or citation_num > len(chunks):
                verification_report.append({
                    "claim": claim["claim"],
                    "citation": citation_num,
                    "supported": False,
                    "reason": "citation number out of range"
                })
                continue

            source_chunk = chunks[citation_num - 1]
            verdict = verify_claim(claim["claim"], source_chunk["text"])

            verification_report.append({
                "claim": claim["claim"],
                "citation": citation_num,
                "supported": verdict.get("supported"),
                "reason": verdict.get("reason")
            })

    return verification_report


if __name__ == "__main__":
    query = "what should I do if a production service goes down"
    result = generate_answer(query)

    print("ANSWER:\n")
    print(result["answer"])

    print("\nCITATION VERIFICATION:\n")
    report = verify_answer(result)
    for entry in report:
        status = "VERIFIED" if entry["supported"] else "UNSUPPORTED"
        print(f"[{status}] citation [{entry['citation']}]: {entry['claim'][:60]}")
        print(f"   reason: {entry['reason']}\n")