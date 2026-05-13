"""
evaluate_fixed_depth_olmes.py
=============================

Run Bolmo fixed-depth quality evaluation with OLMES.

This script evaluates the local fixed-depth-enabled Bolmo HF checkpoint at:

    - 25% global backbone depth
    - 50% global backbone depth
    - 75% global backbone depth
    - 100% global backbone depth

For each depth, it runs both:

    - PIQA
    - HellaSwag

using the Bolmo-specific OLMES backend:

    model_type = "hf_bolmo"

The local OLMES Bolmo wrapper must already support:

    fixed_depth_fraction

and the local HF Bolmo checkpoint must already contain the modified:

    modeling_bolmo.py

with the fixed-depth global-backbone hook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DEPTHS = [0.25, 0.50, 0.75, 1.00]
DEFAULT_TASKS = ["piqa::olmes", "hellaswag::olmes"]


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        required=True,
        help="Local HF Bolmo checkpoint path containing the modified modeling_bolmo.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for all fixed-depth evaluation runs.",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=DEFAULT_DEPTHS,
        help="Depth fractions to evaluate. Default: 0.25 0.50 0.75 1.0",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="OLMES task suites to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="OLMES batch size. Default matches previous Bolmo baseline runs.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum model input length passed to OLMES.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of OLMES workers.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional OLMES --limit value for small debugging runs. Omit for full evaluation.",
    )
    parser.add_argument(
        "--olmes-command",
        default="olmes",
        help="OLMES executable. Default: olmes",
    )

    return parser.parse_args()


# =====================================================================
# Helpers
# =====================================================================

def validate_depths(depths: list[float]) -> list[float]:
    validated = []

    for depth in depths:
        depth = float(depth)

        if not (0.0 < depth <= 1.0):
            raise ValueError(
                f"All depth fractions must be in (0, 1], got {depth}"
            )

        validated.append(depth)

    return validated


def depth_label(depth_fraction: float) -> str:
    """
    Convert a depth fraction to a clean output-folder label.

    Examples:
        0.25 -> "25"
        0.50 -> "50"
        0.75 -> "75"
        1.00 -> "100"
    """
    percentage = depth_fraction * 100.0

    if percentage.is_integer():
        return str(int(percentage))

    return str(percentage).replace(".", "p")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def build_model_args(
    *,
    max_length: int,
    depth_fraction: float,
) -> dict[str, Any]:
    """
    Model arguments passed to OLMES.

    fixed_depth_fraction is handled by the patched
    eleuther_huggingface_bolmo.py wrapper.
    """
    return {
        "max_length": max_length,
        "trust_remote_code": "true",
        "model_type": "hf_bolmo",
        "add_bos_token": "true",
        "fixed_depth_fraction": depth_fraction,
    }


def build_olmes_command(
    *,
    olmes_command: str,
    model_id: str,
    model_args: dict[str, Any],
    tasks: list[str],
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    limit: float | None,
) -> list[str]:
    cmd = [
        olmes_command,
        "--model",
        model_id,
        "--model-args",
        json.dumps(model_args),
        "--task",
        *tasks,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--output-dir",
        str(output_dir),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    return cmd


def command_to_pretty_string(cmd: list[str]) -> str:
    """
    Human-readable command for logs/manifests.
    """
    return " ".join(
        json.dumps(part) if (" " in part or "{" in part or "}" in part) else part
        for part in cmd
    )


# =====================================================================
# Main evaluation loop
# =====================================================================

def main() -> None:
    args = parse_args()
    args.depths = validate_depths(args.depths)

    model_id = str(Path(args.model_id).expanduser())
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_id)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}"
        )

    print("=" * 80)
    print("Bolmo fixed-depth OLMES evaluation")
    print("=" * 80)
    print(f"Model ID / local path: {model_id}")
    print(f"Output directory: {output_dir}")
    print(f"Depths: {args.depths}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Num workers: {args.num_workers}")
    print(f"Limit: {args.limit}")
    print("=" * 80)

    run_start = dt.datetime.now().isoformat()

    global_manifest: dict[str, Any] = {
        "experiment": "fixed_depth_olmes_evaluation",
        "start_time": run_start,
        "model_id": model_id,
        "output_dir": str(output_dir),
        "depths": args.depths,
        "tasks": args.tasks,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "runs": [],
    }

    write_json(output_dir / "fixed_depth_evaluation_config.json", global_manifest)

    for depth_fraction in args.depths:
        label = depth_label(depth_fraction)
        depth_output_dir = output_dir / f"depth_{label}"
        depth_output_dir.mkdir(parents=True, exist_ok=True)

        model_args = build_model_args(
            max_length=args.max_length,
            depth_fraction=depth_fraction,
        )

        cmd = build_olmes_command(
            olmes_command=args.olmes_command,
            model_id=model_id,
            model_args=model_args,
            tasks=args.tasks,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=depth_output_dir,
            limit=args.limit,
        )

        command_str = command_to_pretty_string(cmd)

        run_manifest: dict[str, Any] = {
            "depth_label": label,
            "depth_fraction": depth_fraction,
            "depth_percentage": depth_fraction * 100.0,
            "output_dir": str(depth_output_dir),
            "model_args": model_args,
            "tasks": args.tasks,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "limit": args.limit,
            "command": command_str,
            "start_time": dt.datetime.now().isoformat(),
            "status": "running",
        }

        write_json(depth_output_dir / "depth_evaluation_manifest.json", run_manifest)

        print("\n" + "=" * 80)
        print(f"Starting fixed-depth evaluation: {label}%")
        print(f"Depth fraction: {depth_fraction}")
        print(f"Output directory: {depth_output_dir}")
        print("Command:")
        print(command_str)
        print("=" * 80)

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                text=True,
            )

            run_manifest["status"] = "completed"
            run_manifest["return_code"] = completed.returncode

        except subprocess.CalledProcessError as exc:
            run_manifest["status"] = "failed"
            run_manifest["return_code"] = exc.returncode
            run_manifest["end_time"] = dt.datetime.now().isoformat()

            write_json(
                depth_output_dir / "depth_evaluation_manifest.json",
                run_manifest,
            )

            global_manifest["runs"].append(run_manifest)
            global_manifest["end_time"] = dt.datetime.now().isoformat()
            global_manifest["status"] = "failed"

            write_json(
                output_dir / "fixed_depth_evaluation_summary.json",
                global_manifest,
            )

            print("\n" + "!" * 80)
            print(f"Evaluation failed at depth {label}%")
            print(f"Return code: {exc.returncode}")
            print("!" * 80)

            raise

        run_manifest["end_time"] = dt.datetime.now().isoformat()

        write_json(
            depth_output_dir / "depth_evaluation_manifest.json",
            run_manifest,
        )

        global_manifest["runs"].append(run_manifest)

        print("\n" + "=" * 80)
        print(f"Completed fixed-depth evaluation: {label}%")
        print(f"Results saved to: {depth_output_dir}")
        print("=" * 80)

    global_manifest["end_time"] = dt.datetime.now().isoformat()
    global_manifest["status"] = "completed"

    write_json(
        output_dir / "fixed_depth_evaluation_summary.json",
        global_manifest,
    )

    print("\n" + "=" * 80)
    print("All fixed-depth OLMES evaluations completed successfully.")
    print(f"Summary saved to: {output_dir / 'fixed_depth_evaluation_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)