from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from rageval.evaluation_sharding import build_pair_order, merge_shard_detail_rows
from scripts.evaluate_with_ragas import (
    EvaluationConfig,
    aggregate_rows,
    build_config,
    flatten_detail_rows,
    load_input_rows,
    parse_args as parse_eval_args,
    write_methodology_json,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluate_with_ragas.py in parallel shards with isolated output directories "
            "and merge the shard results into one final evaluation directory."
        )
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of parallel evaluation shards to launch.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("./results/experiment_results_repaired.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Final merged output directory. Defaults to evaluate_with_ragas.py default layout.",
    )
    parser.add_argument("--metric-profile", choices=("deterministic", "balanced", "full"), default="balanced")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--metric-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifest-path", type=Path, default=Path("./dataset/processed/experiment/split_manifest.json"))
    return parser.parse_args()


def build_parallel_config(args: argparse.Namespace) -> EvaluationConfig:
    eval_argv = [
        "--input-csv",
        str(args.input_csv),
        "--metric-profile",
        args.metric_profile,
        "--max-retries",
        str(args.max_retries),
        "--metric-timeout-seconds",
        str(args.metric_timeout_seconds),
    ]
    if args.output_dir is not None:
        eval_argv.extend(["--output-dir", str(args.output_dir)])
    if args.judge_model is not None:
        eval_argv.extend(["--judge-model", args.judge_model])
    if args.embedding_model is not None:
        eval_argv.extend(["--embedding-model", args.embedding_model])
    if args.openai_api_key is not None:
        eval_argv.extend(["--openai-api-key", args.openai_api_key])
    if args.openai_base_url is not None:
        eval_argv.extend(["--openai-base-url", args.openai_base_url])
    if args.limit is not None:
        eval_argv.extend(["--limit", str(args.limit)])
    if args.manifest_path is not None:
        eval_argv.extend(["--manifest-path", str(args.manifest_path)])
    return build_config(parse_eval_args(eval_argv))


def shard_root(output_dir: Path) -> Path:
    return output_dir / "_shards"


def shard_output_dir(output_dir: Path, shard_index: int) -> Path:
    return shard_root(output_dir) / f"shard_{shard_index:02d}"


def shard_log_path(output_dir: Path, shard_index: int) -> Path:
    return shard_root(output_dir) / f"shard_{shard_index:02d}.log"


def launch_shards(config: EvaluationConfig, num_shards: int) -> list[subprocess.Popen[Any]]:
    processes: list[subprocess.Popen[Any]] = []
    script_path = Path(__file__).with_name("evaluate_with_ragas.py")
    shard_root(config.output_dir).mkdir(parents=True, exist_ok=True)

    for shard_index in range(num_shards):
        cmd = [
            sys.executable,
            str(script_path),
            "--input-csv",
            str(config.input_csv),
            "--output-dir",
            str(shard_output_dir(config.output_dir, shard_index)),
            "--metric-profile",
            config.metric_profile,
            "--judge-model",
            config.judge_model,
            "--embedding-model",
            config.embedding_model,
            "--max-retries",
            str(config.max_retries),
            "--metric-timeout-seconds",
            str(config.metric_timeout_seconds),
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_index),
        ]
        if config.openai_api_key:
            cmd.extend(["--openai-api-key", config.openai_api_key])
        if config.openai_base_url:
            cmd.extend(["--openai-base-url", config.openai_base_url])
        if config.limit is not None:
            cmd.extend(["--limit", str(config.limit)])
        if config.manifest_path is not None:
            cmd.extend(["--manifest-path", str(config.manifest_path)])

        log_path = shard_log_path(config.output_dir, shard_index)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_handle.close()
        processes.append(process)
    return processes


def wait_for_shards(processes: list[subprocess.Popen[Any]]) -> None:
    failed: list[tuple[int, int]] = []
    for index, process in enumerate(processes):
        code = process.wait()
        if code != 0:
            failed.append((index, code))
    if failed:
        failures = ", ".join(f"shard {index}: exit {code}" for index, code in failed)
        raise RuntimeError(f"Parallel evaluation failed: {failures}")


def merge_outputs(config: EvaluationConfig, num_shards: int) -> None:
    input_rows = load_input_rows(config.input_csv, limit=config.limit)
    pair_order = build_pair_order(input_rows)
    detail_paths = [
        shard_output_dir(config.output_dir, shard_index) / "evaluation_details.csv"
        for shard_index in range(num_shards)
    ]
    merged_detail_rows = merge_shard_detail_rows(detail_paths, pair_order)
    final_details_csv = config.output_dir / "evaluation_details.csv"
    write_summary_csv(final_details_csv, merged_detail_rows)

    flattened_rows = flatten_detail_rows(merged_detail_rows)
    write_summary_csv(
        config.output_dir / "evaluation_doc_summary.csv",
        aggregate_rows(
            flattened_rows,
            group_fields=[
                "system",
                "dataset_name",
                "doc_id",
                "evaluation_set",
                "length_bucket",
            ],
        ),
    )
    write_summary_csv(
        config.output_dir / "evaluation_bucket_summary.csv",
        aggregate_rows(
            flattened_rows,
            group_fields=["system", "dataset_name", "evaluation_set", "length_bucket"],
        ),
    )
    write_summary_csv(
        config.output_dir / "evaluation_dataset_summary.csv",
        aggregate_rows(
            flattened_rows,
            group_fields=["system", "dataset_name", "evaluation_set"],
        ),
    )
    write_summary_csv(
        config.output_dir / "evaluation_overall_summary.csv",
        aggregate_rows(
            flattened_rows,
            group_fields=["system"],
        ),
    )
    write_methodology_json(config.output_dir / "methodology.json", config)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num_shards must be a positive integer.")
    config = build_parallel_config(args)
    processes = launch_shards(config, args.num_shards)
    wait_for_shards(processes)
    merge_outputs(config, args.num_shards)


if __name__ == "__main__":
    main()
