from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed Assignment 1 evaluation runs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-file", type=Path, default=Path("results/model_comparison.csv")
    )
    parser.add_argument("--include-smoke-tests", action="store_true")
    return parser.parse_args()


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_mean_and_standard_deviation(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def find_training_times(results_directory: Path) -> dict[str, float]:
    training_times: dict[str, float] = {}
    for summary_path in results_directory.rglob("summary.json"):
        summary = load_json(summary_path)
        checkpoint_path = summary.get("best_checkpoint")
        training_time = summary.get("training_time_seconds")
        if checkpoint_path and training_time is not None:
            training_times[str(Path(checkpoint_path).resolve())] = float(training_time)
    return training_times


def read_evaluation_results(
    results_directory: Path,
    include_smoke_tests: bool,
) -> list[dict[str, Any]]:
    training_times = find_training_times(results_directory)
    evaluations: list[dict[str, Any]] = []

    for result_path in sorted(results_directory.rglob("test_results.json")):
        result = load_json(result_path)
        if result.get("smoke_test", False) and not include_smoke_tests:
            continue
        checkpoint_path = str(Path(result["checkpoint"]).resolve())
        result["training_time_seconds"] = training_times.get(checkpoint_path)
        result["result_file"] = str(result_path.resolve())
        evaluations.append(result)
    return evaluations


def deduplicate_evaluations(
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations_by_run: dict[tuple[str, int, str], dict[str, Any]] = {}
    for evaluation in evaluations:
        checkpoint_path = os.path.normcase(
            str(Path(evaluation["checkpoint"]).resolve())
        )
        run_key = (evaluation["model"], int(evaluation["seed"]), checkpoint_path)
        evaluations_by_run[run_key] = evaluation
    return list(evaluations_by_run.values())


def summarize_by_model(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluations = deduplicate_evaluations(evaluations)
    evaluations_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        evaluations_by_model[evaluation["model"]].append(evaluation)

    summaries: list[dict[str, Any]] = []
    for model_name in sorted(evaluations_by_model):
        model_evaluations = evaluations_by_model[model_name]
        accuracies = [item["test_metrics"]["accuracy"] for item in model_evaluations]
        macro_f1_values = [
            item["test_metrics"]["macro_f1"] for item in model_evaluations
        ]
        inference_times = [
            item["inference_timing"]["milliseconds_per_sample"]
            for item in model_evaluations
        ]
        training_times = [
            item["training_time_seconds"]
            for item in model_evaluations
            if item["training_time_seconds"] is not None
        ]

        accuracy_mean, accuracy_std = calculate_mean_and_standard_deviation(accuracies)
        macro_f1_mean, macro_f1_std = calculate_mean_and_standard_deviation(
            macro_f1_values
        )
        inference_mean, inference_std = calculate_mean_and_standard_deviation(
            inference_times
        )
        training_mean, training_std = calculate_mean_and_standard_deviation(
            training_times
        )

        summaries.append(
            {
                "model": model_name,
                "runs": len(model_evaluations),
                "seeds": ",".join(
                    str(seed)
                    for seed in sorted({item["seed"] for item in model_evaluations})
                ),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_std": macro_f1_std,
                "trainable_parameters": model_evaluations[0][
                    "trainable_parameters"
                ],
                "training_time_seconds_mean": training_mean,
                "training_time_seconds_std": training_std,
                "inference_ms_per_sample_mean": inference_mean,
                "inference_ms_per_sample_std": inference_std,
            }
        )
    return summaries


def save_summary_csv(summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    arguments = parse_arguments()
    evaluations = read_evaluation_results(
        arguments.results_dir.resolve(), arguments.include_smoke_tests
    )
    if not evaluations:
        raise FileNotFoundError(
            f"No evaluation results found under {arguments.results_dir.resolve()}"
        )

    summaries = summarize_by_model(evaluations)
    save_summary_csv(summaries, arguments.output_file)
    json_output_path = arguments.output_file.with_suffix(".json")
    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print(json.dumps(summaries, indent=2))
    print(f"Comparison CSV: {arguments.output_file.resolve()}")
    print(f"Comparison JSON: {json_output_path.resolve()}")


if __name__ == "__main__":
    main()
