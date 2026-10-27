from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import load_yaml_file


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed Assignment 1 evaluation runs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-file", type=Path, default=Path("results/model_comparison.csv")
    )
    parser.add_argument(
        "--curves-output",
        type=Path,
        help="Output PNG for the combined training curves (defaults beside --output-file)",
    )
    parser.add_argument("--no-curves", action="store_true")
    parser.add_argument("--include-smoke-tests", action="store_true")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Summarize the GRU/Transformer representation ablation from training summaries.",
    )
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


def read_training_histories(
    results_directory: Path, evaluations: list[dict[str, Any]]
) -> dict[str, dict[int, list[dict[str, float]]]]:
    """Match each evaluated checkpoint to its saved training history."""
    histories: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(dict)
    required_fields = {
        "epoch", "train_loss", "validation_loss", "train_macro_f1", "validation_macro_f1"
    }

    for evaluation in deduplicate_evaluations(evaluations):
        model = str(evaluation["model"])
        seed = int(evaluation["seed"])
        if seed in histories[model]:
            raise ValueError(f"Multiple checkpoints for {model}, seed {seed}; select one run")

        checkpoint = Path(evaluation["checkpoint"])
        run_directory = results_directory / model / checkpoint.parent.name
        summary_path = run_directory / "summary.json"
        history_path = run_directory / "history.csv"
        if not summary_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"Training summary/history missing for {model}, seed {seed}: {run_directory}")
        summary = load_json(summary_path)
        if (
            summary.get("model") != model
            or int(summary.get("seed", -1)) != seed
            or os.path.normcase(str(Path(summary["best_checkpoint"]).resolve()))
            != os.path.normcase(str(checkpoint.resolve()))
        ):
            raise ValueError(f"Training history does not match evaluated checkpoint: {history_path}")

        with history_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not required_fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing curve columns in {history_path}")
            rows = list(reader)
        if not rows or len(rows) != int(summary["epochs_completed"]):
            raise ValueError(f"History length does not match training summary: {history_path}")
        if [int(row["epoch"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Epochs are not contiguous in {history_path}")
        histories[model][seed] = rows

    seed_sets = {frozenset(model_histories) for model_histories in histories.values()}
    if len(seed_sets) != 1:
        raise ValueError("Models have different training seed sets; cannot compare their curves")
    return dict(histories)


def read_representation_ablation_runs(
    results_directory: Path,
    include_smoke_tests: bool,
) -> list[dict[str, Any]]:
    """Read training summaries for the sequence-representation ablation."""
    runs_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    supported_models = {"rnn", "transformer"}
    supported_representations = {
        "image_rows",
        "image_columns",
        "image_patches",
    }

    for summary_path in sorted(results_directory.rglob("summary.json")):
        summary = load_json(summary_path)
        if summary.get("smoke_test", False) and not include_smoke_tests:
            continue

        resolved_config_path = summary_path.parent / "resolved_config.yaml"
        if not resolved_config_path.is_file():
            continue
        resolved_config = load_yaml_file(resolved_config_path)
        model_config = resolved_config.get("model", {})
        model_name = model_config.get("name")
        parameters = model_config.get("parameters", {})
        representation = parameters.get("representation")
        if model_name not in supported_models or representation not in supported_representations:
            continue

        seed = int(summary["seed"])
        key = (str(model_name), str(representation), seed)
        if key in runs_by_key:
            previous = runs_by_key[key]["summary_file"]
            raise ValueError(
                "Multiple ablation runs found for "
                f"{model_name}/{representation}, seed {seed}: {previous} and {summary_path}"
            )

        summary["summary_file"] = str(summary_path.resolve())
        summary["resolved_config_file"] = str(resolved_config_path.resolve())
        summary["variant"] = f"{model_name}/{representation}"
        summary["model"] = str(model_name)
        summary["representation"] = str(representation)
        runs_by_key[key] = summary

    runs = list(runs_by_key.values())
    if not runs:
        raise FileNotFoundError(
            "No GRU/Transformer representation-ablation training summaries found "
            f"under {results_directory.resolve()}"
        )
    return runs


def summarize_representation_ablation(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_variant[run["variant"]].append(run)

    summaries: list[dict[str, Any]] = []
    for variant in sorted(runs_by_variant):
        variant_runs = sorted(runs_by_variant[variant], key=lambda item: int(item["seed"]))
        validation_metrics = [
            run.get("best_validation_metrics") or {} for run in variant_runs
        ]
        macro_f1_values = [float(metrics["macro_f1"]) for metrics in validation_metrics]
        accuracy_values = [float(metrics["accuracy"]) for metrics in validation_metrics]
        loss_values = [float(metrics["loss"]) for metrics in validation_metrics]
        training_times = [float(run["training_time_seconds"]) for run in variant_runs]
        best_epochs = [float(run["best_epoch"]) for run in variant_runs]

        macro_f1_mean, macro_f1_std = calculate_mean_and_standard_deviation(macro_f1_values)
        accuracy_mean, accuracy_std = calculate_mean_and_standard_deviation(accuracy_values)
        loss_mean, loss_std = calculate_mean_and_standard_deviation(loss_values)
        training_mean, training_std = calculate_mean_and_standard_deviation(training_times)
        best_epoch_mean, best_epoch_std = calculate_mean_and_standard_deviation(best_epochs)

        first_run = variant_runs[0]
        summaries.append(
            {
                "model": first_run["model"],
                "representation": first_run["representation"],
                "variant": variant,
                "runs": len(variant_runs),
                "seeds": ",".join(str(run["seed"]) for run in variant_runs),
                "validation_accuracy_mean": accuracy_mean,
                "validation_accuracy_std": accuracy_std,
                "validation_macro_f1_mean": macro_f1_mean,
                "validation_macro_f1_std": macro_f1_std,
                "validation_loss_mean": loss_mean,
                "validation_loss_std": loss_std,
                "trainable_parameters": first_run["trainable_parameters"],
                "best_epoch_mean": best_epoch_mean,
                "best_epoch_std": best_epoch_std,
                "training_time_seconds_mean": training_mean,
                "training_time_seconds_std": training_std,
            }
        )
    return summaries


def read_representation_ablation_histories(
    runs: list[dict[str, Any]],
) -> dict[str, dict[int, list[dict[str, float]]]]:
    """Load and validate one training history per seed and representation."""
    histories: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(dict)
    required_fields = {
        "epoch",
        "train_loss",
        "validation_loss",
        "train_macro_f1",
        "validation_macro_f1",
    }

    for run in runs:
        variant = str(run["variant"])
        seed = int(run["seed"])
        summary_path = Path(run["summary_file"])
        history_path = summary_path.parent / "history.csv"
        if not history_path.is_file():
            raise FileNotFoundError(f"Training history missing: {history_path}")

        with history_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not required_fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing curve columns in {history_path}")
            rows = list(reader)
        if not rows or len(rows) != int(run["epochs_completed"]):
            raise ValueError(f"History length does not match training summary: {history_path}")
        if [int(row["epoch"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Epochs are not contiguous in {history_path}")
        if seed in histories[variant]:
            raise ValueError(f"Multiple histories found for {variant}, seed {seed}")
        histories[variant][seed] = rows

    seed_sets = {frozenset(variant_histories) for variant_histories in histories.values()}
    if len(seed_sets) != 1:
        raise ValueError("Ablation variants have different training seed sets")
    return dict(histories)


def save_summary_csv(summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    arguments = parse_arguments()
    results_directory = arguments.results_dir.resolve()

    if arguments.ablation:
        runs = read_representation_ablation_runs(
            results_directory,
            arguments.include_smoke_tests,
        )
        summaries = summarize_representation_ablation(runs)
        histories = (
            read_representation_ablation_histories(runs)
            if not arguments.no_curves
            else None
        )
        save_summary_csv(summaries, arguments.output_file)
        json_output_path = arguments.output_file.with_suffix(".json")
        with json_output_path.open("w", encoding="utf-8") as file:
            json.dump(summaries, file, indent=2)

        if histories is not None:
            from utils.plots import save_representation_ablation_curves

            curves_output = arguments.curves_output or arguments.output_file.with_name(
                "representation_ablation_curves.png"
            )
            save_representation_ablation_curves(histories, curves_output)
            print(f"Representation ablation curves: {curves_output.resolve()}")

        print(json.dumps(summaries, indent=2))
        print(f"Ablation CSV: {arguments.output_file.resolve()}")
        print(f"Ablation JSON: {json_output_path.resolve()}")
        return

    evaluations = read_evaluation_results(
        results_directory, arguments.include_smoke_tests
    )
    if not evaluations:
        raise FileNotFoundError(
            f"No evaluation results found under {arguments.results_dir.resolve()}"
        )

    summaries = summarize_by_model(evaluations)
    histories = (
        read_training_histories(results_directory, evaluations)
        if not arguments.no_curves else None
    )
    save_summary_csv(summaries, arguments.output_file)
    json_output_path = arguments.output_file.with_suffix(".json")
    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    if histories is not None:
        from utils.plots import save_combined_training_curves

        curves_output = arguments.curves_output or arguments.output_file.with_name(
            "combined_training_curves.png"
        )
        save_combined_training_curves(histories, curves_output)
        print(f"Combined training curves: {curves_output.resolve()}")

    print(json.dumps(summaries, indent=2))
    print(f"Comparison CSV: {arguments.output_file.resolve()}")
    print(f"Comparison JSON: {json_output_path.resolve()}")


if __name__ == "__main__":
    main()
