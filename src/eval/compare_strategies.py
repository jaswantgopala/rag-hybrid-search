import json
from pathlib import Path
from src.eval.run_eval import run_full_eval

STRATEGIES = ["fixed_size", "structure_aware", "semantic"]


def run_comparison(golden_path="data/eval/golden_qa.json", output_path="data/eval/strategy_comparison.json"):
    comparison = {}

    for strategy in STRATEGIES:
        print(f"\n{'='*50}")
        print(f"RUNNING EVAL FOR STRATEGY: {strategy}")
        print(f"{'='*50}\n")

        results_path = f"data/eval/results_{strategy}.json"
        summary = run_full_eval(golden_path=golden_path, strategy=strategy, output_path=results_path)

        comparison[strategy] = {
            "avg_correctness": summary["avg_correctness"],
            "avg_faithfulness": summary["avg_faithfulness"],
            "avg_retrieval_relevance": summary["avg_retrieval_relevance"],
            "avg_citation_accuracy": summary["avg_citation_accuracy"],
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n\n{'='*60}")
    print("STRATEGY COMPARISON REPORT")
    print(f"{'='*60}\n")

    metrics = ["avg_correctness", "avg_faithfulness", "avg_retrieval_relevance", "avg_citation_accuracy"]

    header = f"{'Metric':<28}" + "".join(f"{s:<20}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))

    for metric in metrics:
        row = f"{metric:<28}"
        for strategy in STRATEGIES:
            row += f"{comparison[strategy][metric]:<20}"
        print(row)

    print(f"\nBest per metric:")
    for metric in metrics:
        best_strategy = max(STRATEGIES, key=lambda s: comparison[s][metric])
        best_value = comparison[best_strategy][metric]
        print(f"  {metric}: {best_strategy} ({best_value})")

    print(f"\nFull comparison saved to {output_path}")
    return comparison


if __name__ == "__main__":
    run_comparison()