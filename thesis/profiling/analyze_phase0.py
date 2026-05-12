import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    return p.parse_args()


def load_raw(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else None


def plot_component_breakdown(raw, out_path: Path) -> None:
    comps = ["local_encoder", "boundary", "global_backbone", "local_decoder", "output_head"]
    vals = {c: mean([(r.get("component_times_ms") or {}).get(c) for r in raw]) for c in comps}
    xs = [k for k, v in vals.items() if v is not None]
    ys = [vals[k] for k in xs]
    if not xs:
        print("[warn] no component timings")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys)
    ax.set_xlabel("component")
    ax.set_ylabel("mean time (ms)")
    ax.set_title("Prefill component time breakdown")
    plt.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_global_layer_times(raw, out_path: Path) -> None:
    accum: dict[int, list[float]] = {}
    for r in raw:
        for k, v in (r.get("global_layer_times_ms") or {}).items():
            accum.setdefault(int(k), []).append(float(v))
    if not accum:
        print("[warn] no per-layer timing data")
        return
    xs = sorted(accum)
    ys = [mean(accum[i]) for i in xs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys)
    ax.set_xlabel("global backbone layer index")
    ax.set_ylabel("mean time (ms)")
    ax.set_title("Global Transformer per-layer time (prefill)")
    plt.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_patch_length_dist(raw, out_path: Path) -> None:
    all_lens = []
    for r in raw:
        lens = (r.get("patch_stats") or {}).get("patch_lengths")
        if lens:
            all_lens.extend(float(x) for x in lens)
    if not all_lens:
        print("[warn] no patch lengths to plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_lens, bins=30)
    ax.set_xlabel("patch length (bytes)")
    ax.set_ylabel("count")
    ax.set_title("Patch length distribution")
    plt.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_boundary_confidence_dist(raw, out_path: Path) -> None:
    vals = []
    for r in raw:
        conf = (r.get("patch_stats") or {}).get("boundary_confidences")
        if conf:
            vals.extend(float(x) for x in conf)
    if not vals:
        print("[warn] boundary confidence not available; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=30)
    ax.set_xlabel("boundary confidence")
    ax.set_ylabel("count")
    ax.set_title("Boundary confidence distribution")
    plt.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_latency_vs_patches(raw, out_path: Path) -> None:
    xs, ys = [], []
    for r in raw:
        np_ = (r.get("patch_stats") or {}).get("num_patches")
        pt = r.get("prefill_time_ms")
        if isinstance(np_, (int, float)) and isinstance(pt, (int, float)):
            xs.append(np_); ys.append(pt)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys)
    ax.set_xlabel("num_patches")
    ax.set_ylabel("prefill_time_ms")
    ax.set_title("Prefill latency vs number of patches")
    plt.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    raw = load_raw(in_dir / "raw_results.jsonl")
    if not raw:
        print(f"[error] no records found in {in_dir}")
        return
    plots_dir = in_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_component_breakdown(raw, plots_dir / "component_time_breakdown_prefill.png")
    plot_global_layer_times(raw, plots_dir / "global_layer_time_prefill.png")
    plot_patch_length_dist(raw, plots_dir / "patch_length_distribution.png")
    plot_boundary_confidence_dist(raw, plots_dir / "boundary_confidence_distribution.png")
    plot_latency_vs_patches(raw, plots_dir / "latency_vs_num_patches.png")
    print(f"[info] plots written to {plots_dir}")


if __name__ == "__main__":
    main()
