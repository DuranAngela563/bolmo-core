"""
profile_fixed_depth.py
======================

Phase 1 fixed-depth profiling for Bolmo.

Runs the existing Phase 0 profiling pipeline at several fixed global
backbone depths:

    - 25%
    - 50%
    - 75%
    - 100%

The script reuses the profiling utilities from:

    thesis/profiling/profile_phase0.py

but loads the locally modified BolmoForCausalLM implementation so that
the adaptive-depth hook in modeling_bolmo.py is active.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------
# Make repository paths importable when running this file directly.
# Expected location:
# bolmo-core/thesis/adaptive_depth/fixed_depth/profile_fixed_depth.py
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------
# Local modified Bolmo model: this is essential.
# Do NOT replace this with AutoModelForCausalLM if you want the fixed-depth
# hook in your local modeling_bolmo.py to be used.
# ---------------------------------------------------------------------
from olmo_core.nn.bolmo.hf.modeling_bolmo import BolmoForCausalLM
from olmo_core.nn.bolmo.hf.adaptive_depth import (
    apply_depth_strategy,
    get_active_strategy,
    get_effective_num_layers,
)


# ---------------------------------------------------------------------
# Reuse Phase 0 profiling utilities.
# ---------------------------------------------------------------------
from thesis.profiling.profile_phase0 import (
    free_bolmo_local_caches,
    get_prompt_set,
    git_commit_hash,
    register_timing_hooks,
    resolve_modules,
    run_generation,
    run_prefill,
    save_results,
)


DEFAULT_DEPTHS = [0.25, 0.50, 0.75, 1.00]


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default=os.environ.get("BOLMO_MODEL_ID", "allenai/Bolmo-1B"),
        help="HF model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load local model/tokenizer files.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--prompt-set",
        default="default",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=DEFAULT_DEPTHS,
        help="Fixed-depth fractions to evaluate. Example: --depths 0.25 0.5 0.75 1.0",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("FIXED_DEPTH_OUT_DIR", None),
        help="Base output directory. If omitted, a timestamped directory is created.",
    )
    parser.add_argument(
        "--inspect-modules",
        action="store_true",
    )

    return parser.parse_args()


# =====================================================================
# Helpers
# =====================================================================

def validate_depths(depths: list[float]) -> list[float]:
    validated = []
    for depth in depths:
        if not (0 < depth <= 1.0):
            raise ValueError(
                f"Depth fractions must be in (0, 1], got {depth}"
            )
        validated.append(float(depth))
    return validated


def depth_label(depth_fraction: float) -> str:
    """
    Convert depth fraction into a clean directory label.

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


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT.parent / out_dir
        return out_dir

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    MASTER_THESIS_ROOT = REPO_ROOT.parent

    return (
        MASTER_THESIS_ROOT
        / "outputs"
        / "adaptive_depth"
        / "fixed_depth"
        / f"profiling_{timestamp}"
    )


def load_modified_bolmo_and_tokenizer(args: argparse.Namespace):
    """
    Load the locally modified BolmoForCausalLM class, while fetching weights
    from the supplied HF model ID or local checkpoint path.

    This ensures the modified modeling_bolmo.py with the depth hook is used.
    """
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    print(f"[info] Loading modified Bolmo model from: {args.model_id}")
    print(f"[info] dtype={args.dtype} | device={device}")

    try:
        model = BolmoForCausalLM.from_pretrained(
            args.model_id,
            dtype=dtype,
            local_files_only=args.local_files_only,
        )
    except TypeError:
        model = BolmoForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def apply_depth(model: BolmoForCausalLM, fraction: float) -> dict[str, Any]:
    """
    Apply the requested depth strategy and return clean metadata.

    For 100%, use the explicit "full" strategy so every fixed-depth run
    goes through the same strategy framework.
    """
    if fraction >= 1.0:
        apply_depth_strategy(model.model, "full")
    else:
        apply_depth_strategy(
            model.model,
            "fixed",
            depth_fraction=fraction,
        )

    total_layers = model.model.config.num_hidden_layers
    effective_layers = get_effective_num_layers(model.model)
    strategy_info = get_active_strategy(model.model)

    return {
        "depth_fraction": fraction,
        "depth_percentage": fraction * 100.0,
        "depth_label": depth_label(fraction),
        "strategy": strategy_info,
        "effective_global_layers": effective_layers,
        "total_global_layers": total_layers,
    }


def attach_depth_metadata(
    record: dict[str, Any],
    depth_metadata: dict[str, Any],
    prompt: dict[str, str],
    repeat_idx: int,
) -> dict[str, Any]:
    """
    Merge profiling result dictionaries with experiment-level metadata.
    """
    return {
        "depth_label": depth_metadata["depth_label"],
        "depth_fraction": depth_metadata["depth_fraction"],
        "depth_percentage": depth_metadata["depth_percentage"],
        "depth_strategy": depth_metadata["strategy"]["strategy"],
        "depth_strategy_kwargs": depth_metadata["strategy"]["kwargs"],
        "effective_global_layers": depth_metadata["effective_global_layers"],
        "total_global_layers": depth_metadata["total_global_layers"],
        "repeat_idx": repeat_idx,
        "prompt_id": prompt["id"],
        "prompt_name": prompt["name"],
        **record,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


# =====================================================================
# Per-depth profiling
# =====================================================================

@torch.no_grad()
def profile_one_depth(
    *,
    model: BolmoForCausalLM,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, str]],
    args: argparse.Namespace,
    base_output_dir: Path,
    fraction: float,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the full Phase 0 profiling procedure for one fixed depth.
    """
    depth_meta = apply_depth(model, fraction)
    label = depth_meta["depth_label"]

    depth_output_dir = base_output_dir / f"depth_{label}"
    depth_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(
        f"[depth {label}%] "
        f"strategy={depth_meta['strategy']} | "
        f"layers={depth_meta['effective_global_layers']}/"
        f"{depth_meta['total_global_layers']}"
    )
    print(f"[depth {label}%] output_dir={depth_output_dir}")
    print("=" * 80)

    write_json(depth_output_dir / "depth_metadata.json", depth_meta)

    # -----------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------
    if args.num_warmup > 0:
        print(f"[depth {label}%] Starting warmup: {args.num_warmup} rounds")

    for warmup_idx in range(args.num_warmup):
        print(
            f"[depth {label}%] Warmup "
            f"{warmup_idx + 1}/{args.num_warmup}"
        )

        for prompt in prompts:
            _ = run_prefill(
                model=model,
                tokenizer=tokenizer,
                text=prompt["text"],
                device=device,
                state=state,
            )
            _ = run_generation(
                model=model,
                tokenizer=tokenizer,
                text=prompt["text"],
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

    free_bolmo_local_caches(model)

    # -----------------------------------------------------------------
    # Measured runs
    # -----------------------------------------------------------------
    raw_results: list[dict[str, Any]] = []

    print(f"[depth {label}%] Starting measured runs: {args.num_repeats} repeats")

    for repeat_idx in range(args.num_repeats):
        print(
            f"[depth {label}%] Repeat "
            f"{repeat_idx + 1}/{args.num_repeats}"
        )

        for prompt in prompts:
            print(
                f"    prompt={prompt['id']} "
                f"({prompt['name']})"
            )

            prefill_result = run_prefill(
                model=model,
                tokenizer=tokenizer,
                text=prompt["text"],
                device=device,
                state=state,
            )

            generation_result = run_generation(
                model=model,
                tokenizer=tokenizer,
                text=prompt["text"],
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            combined_result = {
                **prefill_result,
                **generation_result,
            }

            combined_result = attach_depth_metadata(
                record=combined_result,
                depth_metadata=depth_meta,
                prompt=prompt,
                repeat_idx=repeat_idx,
            )

            raw_results.append(combined_result)

    free_bolmo_local_caches(model)

    # -----------------------------------------------------------------
    # Save standard Phase 0-style result files
    # -----------------------------------------------------------------
    save_results(raw_results, depth_output_dir)

    depth_manifest = {
        **depth_meta,
        "output_dir": str(depth_output_dir),
        "num_prompts": len(prompts),
        "num_warmup": args.num_warmup,
        "num_repeats": args.num_repeats,
        "num_raw_rows": len(raw_results),
    }

    write_json(depth_output_dir / "depth_manifest.json", depth_manifest)

    print(
        f"[depth {label}%] Finished. "
        f"Saved {len(raw_results)} measured rows."
    )

    return depth_manifest


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()
    args.depths = validate_depths(args.depths)

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bolmo fixed-depth profiling")
    print("=" * 80)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Depths: {args.depths}")
    print(f"Warmup rounds: {args.num_warmup}")
    print(f"Measured repeats: {args.num_repeats}")
    print(f"Max new tokens: {args.max_new_tokens}")

    model, tokenizer, device = load_modified_bolmo_and_tokenizer(args)

    total_layers = model.model.config.num_hidden_layers
    print(f"[info] Total global backbone layers: {total_layers}")

    prompts = get_prompt_set(args.prompt_set)
    print(f"[info] Loaded {len(prompts)} prompts from prompt set: {args.prompt_set}")

    # -----------------------------------------------------------------
    # Resolve modules and register profiling hooks once.
    # The same hooks are reused across all depths.
    # -----------------------------------------------------------------
    found_names, found_modules, found, missing = resolve_modules(model)

    module_resolution = {
        "found_components": found,
        "missing_components": missing,
        "resolved_module_names": found_names,
    }
    write_json(output_dir / "module_resolution.json", module_resolution)

    if args.inspect_modules:
        print("[info] Module resolution:")
        print(json.dumps(module_resolution, indent=2))

    state, cleanup_hooks = register_timing_hooks(
        found_modules,
        device,
    )

    # -----------------------------------------------------------------
    # Save top-level run config
    # -----------------------------------------------------------------
    run_config = {
        "experiment": "fixed_depth_profiling",
        "timestamp": dt.datetime.now().isoformat(),
        "git_commit": git_commit_hash(),
        "model_id": args.model_id,
        "device": str(device),
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "prompt_set": args.prompt_set,
        "num_prompts": len(prompts),
        "num_warmup": args.num_warmup,
        "num_repeats": args.num_repeats,
        "max_new_tokens": args.max_new_tokens,
        "depths": args.depths,
        "total_global_layers": total_layers,
        "output_dir": str(output_dir),
        "module_resolution": module_resolution,
    }
    write_json(output_dir / "fixed_depth_run_config.json", run_config)

    # -----------------------------------------------------------------
    # Run all fixed depths
    # -----------------------------------------------------------------
    depth_manifests: list[dict[str, Any]] = []

    try:
        for fraction in args.depths:
            manifest = profile_one_depth(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                args=args,
                base_output_dir=output_dir,
                fraction=fraction,
                state=state,
            )
            depth_manifests.append(manifest)
    finally:
        cleanup_hooks()
        free_bolmo_local_caches(model)

    # -----------------------------------------------------------------
    # Save top-level index over all depth runs
    # -----------------------------------------------------------------
    summary_index = {
        "experiment": "fixed_depth_profiling",
        "output_dir": str(output_dir),
        "depth_runs": depth_manifests,
    }
    write_json(output_dir / "fixed_depth_summary_index.json", summary_index)

    print("\n" + "=" * 80)
    print("Fixed-depth profiling complete")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()