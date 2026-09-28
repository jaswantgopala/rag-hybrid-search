import json
import re
import time
from pathlib import Path
from src.generation.graceful_fallback import answer_with_fallback
from src.generation.local_llm import local_chat_json


def call_with_retry(fn, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                match = re.search(r"try again in (\d+)m([\d.]+)s", error_str)
                if match:
                    wait_seconds = int(match.group(1)) * 60 + float(match.group(2)) + 5
                else:
                    wait_seconds = 60
                print(f"  rate limited, waiting {wait_seconds:.0f}s before retry...")
                time.sleep(wait_seconds)
            else:
                raise


def llm_judge_correctness(question, golden_answer, generated_answer):
    if golden_answer == "NO_ANSWER":
        said_no_answer = any(
            phrase in generated_answer.lower()
            for phrase in ["couldn't find", "cannot find", "not enough information", "don't have", "no information"]
        )
        return 1.0 if said_no_answer else 0.0

    prompt = f"""Question: {question}

Golden (reference) answer: {golden_answer}

Generated answer: {generated_answer}

On a scale of 0.0 to 1.0, how correct is the generated answer compared to the golden answer?
Respond with only valid JSON: {{"score": a number from 0.0 to 1.0, "reason": "one short sentence"}}"""

    result = local_chat_json(prompt, fallback={"score": 0.0})
    return float(result.get("score", 0.0))


def retrieval_relevance(expected_sources, retrieved_sources):
    if not expected_sources:
        return 1.0
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    overlap = retrieved_set.intersection(expected_set)
    return len(overlap) / len(expected_set)


def run_single_eval(qa_item, strategy="structure_aware"):
    question = qa_item["question"]
    golden_answer = qa_item["expected_answer"]
    expected_sources = qa_item["expected_sources"]

    result = call_with_retry(answer_with_fallback, question, strategy=strategy)

    generated_answer = result["answer"]
    retrieved_sources = result.get("sources", [])
    confidence_scores = result["confidence_scores"]

    correctness = llm_judge_correctness(question, golden_answer, generated_answer)
    relevance = retrieval_relevance(expected_sources, retrieved_sources)

    faithfulness = confidence_scores["citation_coverage"]["verified_citation_ratio"] if result["status"] == "answered" else 1.0
    citation_accuracy = faithfulness

    return {
        "id": qa_item["id"],
        "question": question,
        "type": qa_item["type"],
        "status": result["status"],
        "correctness": round(correctness, 3),
        "faithfulness": round(faithfulness, 3),
        "retrieval_relevance": round(relevance, 3),
        "citation_accuracy": round(citation_accuracy, 3),
        "composite_confidence": confidence_scores["composite_score"],
    }


def run_full_eval(golden_path="data/eval/golden_qa.json", strategy="structure_aware", output_path="data/eval/results.json"):
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    results = []
    for i, qa_item in enumerate(golden_qa, start=1):
        print(f"[{i}/{len(golden_qa)}] running: {qa_item['question'][:60]}")
        try:
            eval_result = run_single_eval(qa_item, strategy=strategy)
            results.append(eval_result)
        except Exception as e:
            print(f"  failed: {e}")
            results.append({"id": qa_item["id"], "question": qa_item["question"], "error": str(e)})

    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("\nAll questions failed - no summary to compute.")
        summary = {
            "strategy": strategy,
            "total_questions": len(golden_qa),
            "successful": 0,
            "results": results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    avg_correctness = sum(r["correctness"] for r in valid_results) / len(valid_results)
    avg_faithfulness = sum(r["faithfulness"] for r in valid_results) / len(valid_results)
    avg_relevance = sum(r["retrieval_relevance"] for r in valid_results) / len(valid_results)
    avg_citation_accuracy = sum(r["citation_accuracy"] for r in valid_results) / len(valid_results)

    summary = {
        "strategy": strategy,
        "total_questions": len(golden_qa),
        "avg_correctness": round(avg_correctness, 3),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_retrieval_relevance": round(avg_relevance, 3),
        "avg_citation_accuracy": round(avg_citation_accuracy, 3),
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== EVAL SUMMARY ({strategy}) ===")
    print(f"Avg correctness: {summary['avg_correctness']}")
    print(f"Avg faithfulness: {summary['avg_faithfulness']}")
    print(f"Avg retrieval relevance: {summary['avg_retrieval_relevance']}")
    print(f"Avg citation accuracy: {summary['avg_citation_accuracy']}")
    print(f"\nFull results saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_full_eval()