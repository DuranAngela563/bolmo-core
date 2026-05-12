import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


COMPONENT_MODULES = {
    "local_encoder": ["model.local_encoder", "local_encoder"],
    "boundary": ["model.local_encoder.boundary_predictor_module", "local_encoder.boundary_predictor_module"],
    "global_backbone": ["model.layers", "layers"],
    "local_decoder": ["model.local_decoder", "local_decoder"],
    "output_head": ["lm_head", "model.lm_head"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="allenai/Bolmo-1B")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-repeats", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--prompt-set", default="default")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--inspect-modules", action="store_true")
    return p.parse_args()


def get_prompt_set(_: str) -> list[dict[str, str]]:
    return [
        {"id": "short", "name": "short_sentence", "text": "Summarize this model's purpose in one sentence."},
        {
            "id": "medium",
            "name": "medium_paragraph",
            "text": (
                "Byte-level language models remove tokenization but still depend on local-to-global "
                "compression. Explain why profiling local encoder, patching, and global Transformer "
                "time is useful before adaptive depth."
            ),
        },
        {
            "id": "long",
            "name": "long_multiparagraph",
            "text": (
                "We are investigating LayerSkip-style adaptive depth for tokenizer-free byte models.\n\n"
                "Before routing changes, we need a baseline of latency and memory breakdown. Measure "
                "prefill and generation, and estimate cache usage.\n\n"
                "If patch and boundary signals are available, relate them to latency so we can decide "
                "whether the global backbone is the primary bottleneck."
            ),
        },
        {
            "id": "code",
            "name": "code_like",
            "text": (
                "def rolling_hash(data: bytes, base=257, mod=2**61-1):\n"
                "    h = 0\n"
                "    for b in data:\n"
                "        h = (h * base + b) % mod\n"
                "    return h"
            ),
        },
        {"id": "noisy", "name": "noisy_typo", "text": "plz explain whhy byte-level models stay robust to weird punctuashun??"},
        {
            "id": "long_1k",
            "name": "long_1k_bytes",
            "text": (
                "The Transformer architecture has become the dominant paradigm in natural language processing, "
                "powering models from BERT to GPT-4. At its core, the self-attention mechanism allows each "
                "position in a sequence to attend to all other positions, creating rich contextual representations. "
                "However, the quadratic cost of attention with respect to sequence length has motivated numerous "
                "efficiency improvements.\n\n"
                "Byte-level models take this further by operating directly on raw bytes rather than subword tokens. "
                "This eliminates the tokenization step entirely, making models robust to typos, code, multilingual "
                "text, and novel character sequences. The trade-off is that byte sequences are roughly 4x longer "
                "than their token equivalents, making the quadratic attention cost even more problematic.\n\n"
                "Hierarchical byte models like BLT, HAT, and Bolmo address this by introducing a two-level architecture: "
                "a local encoder that compresses bytes into patches, and a global transformer that operates on these "
                "compressed patch representations. The boundary between local and global processing is determined by "
                "a segmentation strategy that varies across architectures."
            ),
        },
        {
            "id": "long_2k",
            "name": "long_2k_bytes",
            "text": (
                "In the field of machine learning, the challenge of efficient inference has become increasingly "
                "important as models grow larger. The key insight behind adaptive computation is that not all inputs "
                "require the same amount of processing. Simple inputs like common words or punctuation can be handled "
                "by early layers, while complex or ambiguous inputs may need the full depth of the network.\n\n"
                "Early exit strategies allow a model to terminate processing at an intermediate layer if a confidence "
                "threshold is met. This has been explored extensively for token-level transformer models, where approaches "
                "like CALM, LayerSkip, and DeeBERT have demonstrated significant speedups with minimal quality degradation. "
                "The core mechanism involves training lightweight exit classifiers at each layer that predict whether the "
                "current representation is sufficient for accurate output.\n\n"
                "For byte-level hierarchical models, adaptive depth presents unique opportunities. The two-level architecture "
                "creates natural points for adaptive computation at both the patch level and the byte level. A patch-level "
                "router could decide how many global Transformer layers are needed for each compressed patch, while a byte-level "
                "router could decide whether all local decoder layers are necessary for predictable bytes.\n\n"
                "Consider the implications: a common English word like 'the' compressed into a single patch might only need "
                "part of the backbone, while a code snippet with unusual syntax might need all layers. The challenge is designing "
                "a routing mechanism that is cheap relative to the computation it saves. The profiling phase quantifies these "
                "trade-offs by measuring component costs, per-layer costs, memory use, cache size, and the relation between input "
                "structure and processing cost."
            ),
        },


                {
            "id": "long_4k",
            "name": "long_4k_bytes",
            "text": (
                "Tokenizer-free language models are attractive because they remove the fixed vocabulary used by "
                "standard subword tokenizers. Instead of mapping text into learned word pieces, they operate closer "
                "to the raw byte or character sequence. This makes them naturally robust to spelling mistakes, rare "
                "words, code snippets, multilingual text, and unusual formatting. However, this robustness comes with "
                "a cost: byte sequences are much longer than token sequences, so directly applying a full Transformer "
                "over bytes would be very expensive.\n\n"

                "Hierarchical byte models address this problem by separating local and global computation. A local "
                "encoder first processes the byte stream and groups nearby bytes into patches. These patches are then "
                "processed by a global Transformer backbone. Finally, a local decoder maps the global patch-level "
                "representations back to byte-level predictions. This design reduces the effective sequence length "
                "seen by the global Transformer, while still allowing the model to produce byte-level outputs.\n\n"

                "For profiling, it is important to understand how much time is spent in each part of the architecture. "
                "If the global Transformer dominates, then adaptive global depth is a natural target: easy patches could "
                "use fewer global layers, while difficult patches could use the full backbone. If the local decoder is "
                "also expensive, then byte-level adaptive depth may become important as well. The goal of Phase 0 is not "
                "to implement routing yet, but to measure where the computational cost actually appears.\n\n"

                "Patch statistics are especially important in this setting. Two inputs with the same number of bytes can "
                "produce very different numbers of patches depending on their structure. Natural language may compress "
                "well because common words and predictable character sequences are grouped into longer patches. Code or "
                "noisy text may create shorter patches because punctuation, indentation, identifiers, and unusual symbols "
                "make the boundary predictor split more often. As a result, patch count can be a better predictor of KV "
                "cache size and global backbone cost than raw byte length.\n\n"

                "The adaptive-depth idea is inspired by LayerSkip-style inference, where not every input needs the full "
                "depth of the model. In a hierarchical byte model, the routing decision could use hierarchy-native signals "
                "such as patch length, boundary confidence, local uncertainty, or decoder entropy. A long and confident "
                "patch corresponding to a common word may require fewer layers, while a short or uncertain patch in a code "
                "fragment may require more computation. The challenge is to make routing cheap enough that the saved layer "
                "compute is larger than the routing overhead.\n\n"

                "This profiling experiment should therefore measure prefill time, generation time, peak memory, KV cache "
                "size, component-level timing, per-layer backbone timing, local decoder timing, patch count, patch length "
                "distribution, and boundary confidence distribution. These metrics make it possible to decide whether the "
                "global backbone is the main bottleneck, whether the local decoder must also be considered, and whether "
                "patch-level signals are informative enough to support adaptive computation.\n\n"

                "A useful outcome of this phase would be a clear table showing how latency and memory change as the input "
                "gets longer. If the global backbone percentage increases with patch count, then adaptive global depth is "
                "strongly motivated for long-context inference. If the local decoder grows faster, then the thesis should "
                "frame global adaptive depth as one part of a broader hierarchy-aware efficiency strategy. If memory grows "
                "linearly with patch count, then patch-aware cache analysis also becomes important for understanding the "
                "inference behavior of tokenizer-free models.\n\n"
            ),
        },
    ]


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_peak_memory_start(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def measure_peak_memory_end(device: torch.device) -> tuple[Optional[float], Optional[float]]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2), torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return None, None


def memory_allocated_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return None


def summarize_list(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {k: None for k in ["mean", "std", "min", "max", "p50", "p90", "p95"]}
    s = sorted(values)

    def idx(q: float) -> float:
        return s[min(len(s) - 1, int(q * (len(s) - 1)))]

    return {
        "mean": statistics.mean(s),
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min": min(s),
        "max": max(s),
        "p50": idx(0.5),
        "p90": idx(0.9),
        "p95": idx(0.95),
    }


def estimate_tensor_bytes(obj: Any) -> Optional[float]:
    seen: set[int] = set()

    def rec(x: Any) -> int:
        if id(x) in seen:
            return 0
        seen.add(id(x))
        if torch.is_tensor(x):
            return x.numel() * x.element_size()
        if isinstance(x, dict):
            return sum(rec(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return sum(rec(v) for v in x)
        if hasattr(x, "__dict__"):
            return rec(vars(x))
        return 0

    total = rec(obj)
    return total / (1024 ** 2) if total > 0 else None


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=args.local_files_only,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def resolve_modules(model: torch.nn.Module) -> tuple[dict[str, Optional[str]], dict[str, Optional[torch.nn.Module]], list[str], list[str]]:
    named = dict(model.named_modules())
    found_names: dict[str, Optional[str]] = {}
    found_modules: dict[str, Optional[torch.nn.Module]] = {}
    found: list[str] = []
    missing: list[str] = []
    for comp, candidates in COMPONENT_MODULES.items():
        name = next((c for c in candidates if c in named), None)
        if name is None:
            print(f"[warn] could not resolve module for component={comp}")
            missing.append(comp)
        else:
            found.append(comp)
        found_names[comp] = name
        found_modules[comp] = named.get(name) if name else None
    return found_names, found_modules, found, missing


def free_bolmo_local_caches(model) -> None:
    inner = getattr(model, "model", None)
    if inner is None:
        return
    local_encoder = getattr(inner, "local_encoder", None)
    if local_encoder is not None and hasattr(local_encoder, "free_inference_cache"):
        local_encoder.free_inference_cache()
    local_decoder = getattr(inner, "local_decoder", None)
    if local_decoder is not None and hasattr(local_decoder, "free_inference_cache"):
        local_decoder.free_inference_cache()


def _capture_local_encoder_output(state: dict[str, Any], output: Any) -> None:
    if not isinstance(output, (tuple, list)) or len(output) < 4:
        state["captured"]["local_encoder_output_type"] = type(output).__name__
        return
    boundary_logprobs = output[2]
    boundary_mask = output[3]
    if torch.is_tensor(boundary_logprobs):
        state["captured"]["boundary_logprobs"] = boundary_logprobs.detach().cpu()
        state["captured"]["boundary_logprobs_shape"] = tuple(boundary_logprobs.shape)
    if torch.is_tensor(boundary_mask):
        state["captured"]["boundary_mask"] = boundary_mask.detach().cpu()
        state["captured"]["boundary_mask_shape"] = tuple(boundary_mask.shape)


def register_timing_hooks(modules: dict[str, Optional[torch.nn.Module]], device: torch.device):
    state: dict[str, Any] = {
        "enabled": False,
        "components": defaultdict(float),
        "global_layers": defaultdict(float),
        "decoder_layers": defaultdict(float),
        "hooks_fired": set(),
        "captured": {},
    }
    hooks = []

    def timed(name: str, bucket: str):
        start = {"t": 0.0}

        def pre(*_):
            if not state["enabled"]:
                return
            cuda_sync(device)
            start["t"] = time.perf_counter()

        def post(module, inputs, output):
            if not state["enabled"]:
                return
            cuda_sync(device)
            state[bucket][name] += (time.perf_counter() - start["t"]) * 1000.0
            state["hooks_fired"].add(f"{bucket}/{name}")
            if bucket == "components" and name == "local_encoder":
                _capture_local_encoder_output(state, output)

        return pre, post

    for comp in ["local_encoder", "boundary", "global_backbone", "local_decoder", "output_head"]:
        mod = modules.get(comp)
        if mod is None:
            continue
        pre, post = timed(comp, "components")
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))

    gl = modules.get("global_backbone")
    if isinstance(gl, torch.nn.ModuleList):
        for i, layer in enumerate(gl):
            pre, post = timed(str(i), "global_layers")
            hooks.append(layer.register_forward_pre_hook(pre))
            hooks.append(layer.register_forward_hook(post))
        print(f"[info] hooked {len(gl)} global_backbone layers")
    elif gl is not None:
        print(f"[warn] global_backbone is {type(gl).__name__}, not ModuleList; only container hook registered")

    dec = modules.get("local_decoder")
    if dec is not None:
        hooked_decoder_parts: list[str] = []
        for name in ["initial_norm", "in_projection", "out_norm", "out_projection"]:
            child = getattr(dec, name, None)
            if isinstance(child, torch.nn.Module):
                pre, post = timed(name, "decoder_layers")
                hooks.append(child.register_forward_pre_hook(pre))
                hooks.append(child.register_forward_hook(post))
                hooked_decoder_parts.append(name)
        layers = getattr(dec, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            for i, layer in enumerate(layers):
                lname = f"layers.{i}"
                pre, post = timed(lname, "decoder_layers")
                hooks.append(layer.register_forward_pre_hook(pre))
                hooks.append(layer.register_forward_hook(post))
                hooked_decoder_parts.append(lname)
        if hooked_decoder_parts:
            print(f"[info] hooked local_decoder parts: {hooked_decoder_parts}")
        else:
            print("[warn] no local_decoder child/layer hooks registered")
    else:
        print("[warn] local_decoder module not found")

    def cleanup():
        for h in hooks:
            h.remove()

    return state, cleanup


def _fill_patch_stats_from_mask(out: dict[str, Any], boundary_mask: torch.Tensor, input_ids: torch.Tensor) -> None:
    b = boundary_mask.detach()
    if b.ndim > 1:
        b = b[0]
    b = b.bool().cpu()
    idx = torch.where(b)[0].tolist()
    lengths: list[int] = []
    prev = 0
    seq_len = int(input_ids.shape[1])
    for x in idx:
        lengths.append(int(x - prev + 1))
        prev = int(x + 1)
    if prev < seq_len:
        lengths.append(int(seq_len - prev))
    out["num_patches"] = len(lengths)
    out["patch_lengths"] = lengths
    out["input_units_per_patch"] = (seq_len / len(lengths)) if lengths else None
    out["patch_length_stats"] = summarize_list([float(x) for x in lengths])


def _fill_confidence_stats_from_logprobs(out: dict[str, Any], boundary_logprobs: torch.Tensor) -> None:
    bp = boundary_logprobs.detach().float()
    if bp.ndim > 1:
        bp = bp[0]
    if bp.ndim == 1:
        conf_tensor = torch.exp(bp).clamp(0, 1)
        out["boundary_confidence_source"] = "captured_local_encoder_boundary_logprobs_exp_1d"
    elif bp.ndim == 2 and bp.shape[-1] == 2:
        conf_tensor = torch.exp(bp[:, 1]).clamp(0, 1)
        out["boundary_confidence_source"] = "captured_local_encoder_boundary_logprobs_exp_class1"
    else:
        print(f"[warn] unexpected boundary_logprobs shape: {tuple(boundary_logprobs.shape)}")
        return
    conf = conf_tensor.cpu().tolist()
    out["boundary_confidences"] = conf
    out["boundary_confidences_available"] = True
    out["boundary_confidence_stats"] = summarize_list([float(x) for x in conf])


def make_patch_stats_from_captured(captured: dict[str, Any], input_ids: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_patches": None,
        "input_units_per_patch": None,
        "patch_lengths": None,
        "patch_length_stats": summarize_list([]),
        "boundary_confidences": None,
        "boundary_confidence_stats": summarize_list([]),
        "boundary_confidences_available": False,
        "patch_stats_source": "unavailable",
        "boundary_confidence_source": "unavailable",
        "boundary_mask_shape": captured.get("boundary_mask_shape"),
        "boundary_logprobs_shape": captured.get("boundary_logprobs_shape"),
    }
    boundary_mask = captured.get("boundary_mask")
    boundary_logprobs = captured.get("boundary_logprobs")
    if torch.is_tensor(boundary_mask):
        _fill_patch_stats_from_mask(out, boundary_mask, input_ids)
        out["patch_stats_source"] = "captured_local_encoder_boundary_mask"
    if torch.is_tensor(boundary_logprobs):
        _fill_confidence_stats_from_logprobs(out, boundary_logprobs)
    return out


def fallback_patch_stats_from_prefill_boundary_forward(model, input_ids: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_patches": None,
        "input_units_per_patch": None,
        "patch_lengths": None,
        "patch_length_stats": summarize_list([]),
        "boundary_confidences": None,
        "boundary_confidence_stats": summarize_list([]),
        "boundary_confidences_available": False,
        "patch_stats_source": "unavailable",
        "boundary_confidence_source": "unavailable",
        "boundary_mask_shape": None,
        "boundary_logprobs_shape": None,
    }
    try:
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "prefill_boundary_prediction_forward"):
            return out
        free_bolmo_local_caches(model)
        boundary_mask = inner.prefill_boundary_prediction_forward(input_ids)
        if torch.is_tensor(boundary_mask):
            out["boundary_mask_shape"] = tuple(boundary_mask.shape)
            _fill_patch_stats_from_mask(out, boundary_mask.detach().cpu(), input_ids)
            out["patch_stats_source"] = "model.prefill_boundary_prediction_forward"
        free_bolmo_local_caches(model)
    except Exception as e:
        print(f"[warn] fallback prefill_boundary_prediction_forward failed: {e}")
    return out


def run_prefill(model, tokenizer, text: str, device: torch.device, state: dict[str, Any]) -> dict[str, Any]:
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]

    free_bolmo_local_caches(model)
    state["enabled"] = True
    state["components"].clear()
    state["global_layers"].clear()
    state["decoder_layers"].clear()
    state["hooks_fired"].clear()
    state["captured"].clear()

    before_alloc = memory_allocated_mb(device)
    measure_peak_memory_start(device)
    cuda_sync(device)
    t0 = time.perf_counter()
    try:
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        cuda_sync(device)
        t_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        state["enabled"] = False

    peak_alloc, peak_reserved = measure_peak_memory_end(device)
    component_times = dict(state["components"])
    layer_times = dict(state["global_layers"])
    decoder_layer_times = dict(state["decoder_layers"])
    hooks_fired = sorted(state["hooks_fired"])
    captured = dict(state["captured"])

    gb_total = component_times.get("global_backbone")
    if gb_total is not None:
        gb_source = "component_hook"
    elif layer_times:
        gb_total = sum(layer_times.values())
        component_times["global_backbone"] = gb_total
        gb_source = "sum_global_layer_hooks"
    else:
        gb_total = None
        gb_source = "unavailable"

    dec_total = component_times.get("local_decoder")
    if dec_total is not None:
        dec_source = "component_hook"
    elif decoder_layer_times:
        dec_total = sum(decoder_layer_times.values())
        component_times["local_decoder"] = dec_total
        dec_source = "sum_decoder_layer_hooks"
    else:
        dec_total = None
        dec_source = "unavailable"

    top_level_keys = ["local_encoder", "global_backbone", "local_decoder", "output_head"]
    top_level_hooked_total = sum(component_times.get(k, 0.0) for k in top_level_keys)
    raw_all_hooked_sum = sum(component_times.values())
    unaccounted_ms = t_ms - top_level_hooked_total

    patch_stats = make_patch_stats_from_captured(captured, input_ids)
    if patch_stats.get("num_patches") is None:
        fallback = fallback_patch_stats_from_prefill_boundary_forward(model, input_ids)
        if fallback.get("num_patches") is not None:
            patch_stats = fallback

    return {
        "input_bytes": len(text.encode("utf-8")),
        "input_sequence_length": int(input_ids.shape[1]),
        "prefill_time_ms": t_ms,
        "prefill_peak_memory_allocated_mb": peak_alloc,
        "prefill_peak_memory_reserved_mb": peak_reserved,
        "prefill_peak_memory_delta_allocated_mb": (peak_alloc - before_alloc) if (peak_alloc is not None and before_alloc is not None) else None,
        "prefill_kv_cache_size_mb": estimate_tensor_bytes(getattr(outputs, "past_key_values", None)),
        "component_times_ms": component_times,
        "global_layer_times_ms": layer_times,
        "decoder_layer_times_ms": decoder_layer_times,
        "global_backbone_total_time_ms": gb_total,
        "global_backbone_time_source": gb_source,
        "local_decoder_total_time_ms": dec_total,
        "local_decoder_time_source": dec_source,
        "top_level_hooked_time_ms": top_level_hooked_total,
        "raw_all_hooked_sum_ms": raw_all_hooked_sum,
        "unaccounted_time_ms": unaccounted_ms,
        "hooks_fired": hooks_fired,
        "captured_boundary_mask_shape": captured.get("boundary_mask_shape"),
        "captured_boundary_logprobs_shape": captured.get("boundary_logprobs_shape"),
        "patch_stats": patch_stats,
    }


def run_generation(model, tokenizer, text: str, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]

    before_alloc = memory_allocated_mb(device)
    measure_peak_memory_start(device)
    cuda_sync(device)
    t0 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    cuda_sync(device)
    t_ms = (time.perf_counter() - t0) * 1000.0
    peak_alloc, peak_reserved = measure_peak_memory_end(device)
    free_bolmo_local_caches(model)

    full_text = tokenizer.decode(out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if full_text.startswith(prompt_text):
        continuation_text = full_text[len(prompt_text):]
        continuation_source = "full_text_minus_decoded_prompt"
    elif full_text.startswith(text):
        continuation_text = full_text[len(text):]
        continuation_source = "full_text_minus_original_text"
    else:
        continuation_text = ""
        continuation_source = "unmatched_prefix"

    generated_bytes = len(continuation_text.encode("utf-8")) if continuation_text else 0
    if generated_bytes > 0:
        tpb = t_ms / generated_bytes
        bps = generated_bytes / (t_ms / 1000.0)
    else:
        tpb = None
        bps = None

    return {
        "decode_mode": "generate_total_fallback",
        "generate_total_time_ms": t_ms,
        "generate_total_time_per_generated_byte_ms": tpb,
        "generated_bytes": generated_bytes,
        "generate_total_bytes_per_second": bps,
        "generate_total_peak_memory_allocated_mb": peak_alloc,
        "generate_total_peak_memory_reserved_mb": peak_reserved,
        "generate_total_peak_memory_delta_allocated_mb": (peak_alloc - before_alloc) if (peak_alloc is not None and before_alloc is not None) else None,
        "generate_total_kv_cache_size_mb": None,
        "generation_continuation_source": continuation_source,
        "output_sequence_length": int(out.shape[1]) if hasattr(out, "shape") else None,
        "generation_debug_prompt_text": prompt_text[:300],
        "generation_debug_full_text_prefix": full_text[:300],
        "generation_debug_continuation": continuation_text[:300],
    }


def git_commit_hash() -> Optional[str]:
    out = subprocess.getoutput("git rev-parse HEAD").strip()
    return out if out and "fatal:" not in out else None


def _mean(vals: list[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _std(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def save_results(raw: list[dict[str, Any]], out_dir: Path) -> None:
    with (out_dir / "raw_results.jsonl").open("w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        grouped[r["prompt_id"]].append(r)

    summary = []
    for pid, rows in grouped.items():
        def vals(key: str) -> list[float]:
            return [x[key] for x in rows if isinstance(x.get(key), (int, float))]

        def patch_vals(key: str) -> list[float]:
            out = []
            for x in rows:
                v = (x.get("patch_stats") or {}).get(key)
                if isinstance(v, (int, float)):
                    out.append(v)
            return out

        summary.append({
            "prompt_id": pid,
            "prompt_name": rows[0].get("prompt_name"),
            "input_bytes_mean": _mean(vals("input_bytes")),
            "input_sequence_length_mean": _mean(vals("input_sequence_length")),
            "generated_bytes_mean": _mean(vals("generated_bytes")),
            "prefill_time_ms_mean": _mean(vals("prefill_time_ms")),
            "prefill_time_ms_std": _std(vals("prefill_time_ms")),
            "generate_total_time_ms_mean": _mean(vals("generate_total_time_ms")),
            "generate_total_time_ms_std": _std(vals("generate_total_time_ms")),
            "generate_total_time_per_generated_byte_ms_mean": _mean(vals("generate_total_time_per_generated_byte_ms")),
            "generate_total_bytes_per_second_mean": _mean(vals("generate_total_bytes_per_second")),
            "prefill_peak_memory_allocated_mb_mean": _mean(vals("prefill_peak_memory_allocated_mb")),
            "prefill_peak_memory_reserved_mb_mean": _mean(vals("prefill_peak_memory_reserved_mb")),
            "prefill_peak_memory_delta_allocated_mb_mean": _mean(vals("prefill_peak_memory_delta_allocated_mb")),
            "prefill_kv_cache_size_mb_mean": _mean(vals("prefill_kv_cache_size_mb")),
            "generate_total_peak_memory_allocated_mb_mean": _mean(vals("generate_total_peak_memory_allocated_mb")),
            "generate_total_peak_memory_reserved_mb_mean": _mean(vals("generate_total_peak_memory_reserved_mb")),
            "generate_total_peak_memory_delta_allocated_mb_mean": _mean(vals("generate_total_peak_memory_delta_allocated_mb")),
            "global_backbone_total_time_ms_mean": _mean(vals("global_backbone_total_time_ms")),
            "global_backbone_time_source": rows[0].get("global_backbone_time_source"),
            "global_backbone_fraction_of_prefill_mean": _mean(vals("global_backbone_fraction_of_prefill")),
            "local_decoder_total_time_ms_mean": _mean(vals("local_decoder_total_time_ms")),
            "local_decoder_time_source": rows[0].get("local_decoder_time_source"),
            "local_decoder_fraction_of_prefill_mean": _mean(vals("local_decoder_fraction_of_prefill")),
            "top_level_hooked_time_ms_mean": _mean(vals("top_level_hooked_time_ms")),
            "raw_all_hooked_sum_ms_mean": _mean(vals("raw_all_hooked_sum_ms")),
            "unaccounted_time_ms_mean": _mean(vals("unaccounted_time_ms")),
            "num_patches_mean": _mean(patch_vals("num_patches")),
            "input_units_per_patch_mean": _mean(patch_vals("input_units_per_patch")),
            "patch_stats_source": (rows[0].get("patch_stats") or {}).get("patch_stats_source"),
            "boundary_confidences_available": (rows[0].get("patch_stats") or {}).get("boundary_confidences_available"),
            "boundary_confidence_source": (rows[0].get("patch_stats") or {}).get("boundary_confidence_source"),
            "generation_continuation_source": rows[0].get("generation_continuation_source"),
        })

    fieldnames = list(summary[0].keys()) if summary else ["prompt_id"]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"outputs/profiling/phase0_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading model: {args.model_id}")
    model, tokenizer, device = load_model_and_tokenizer(args)
    print(f"[info] device={device}, dtype={args.dtype}")

    component_paths, modules, found_components, missing_components = resolve_modules(model)
    if args.inspect_modules:
        for k, v in component_paths.items():
            print(f"[inspect] {k}: {v}")
        print("[inspect] relevant named_modules:")
        shown = 0
        for name, _ in model.named_modules():
            lname = name.lower()
            if any(k in lname for k in ["encoder", "decoder", "boundary", "layer", "lm_head"]):
                print(f"  {name}")
                shown += 1
                if shown >= 200:
                    print("[inspect] stopped after 200 matching modules")
                    break

    state, remove_hooks = register_timing_hooks(modules, device)
    num_layers = len(modules["global_backbone"]) if isinstance(modules.get("global_backbone"), torch.nn.ModuleList) else None
    dec = modules.get("local_decoder")
    local_decoder_layers = len(getattr(dec, "layers")) if dec is not None and isinstance(getattr(dec, "layers", None), torch.nn.ModuleList) else None

    metadata = {
        "model_id": args.model_id,
        "datetime": dt.datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "num_warmup": args.num_warmup,
        "num_repeats": args.num_repeats,
        "max_new_tokens": args.max_new_tokens,
        "prompt_set": args.prompt_set,
        "components_found": found_components,
        "components_missing": missing_components,
        "component_modules": component_paths,
        "global_layers_path": component_paths.get("global_backbone"),
        "num_global_layers": num_layers,
        "num_local_decoder_layers": local_decoder_layers,
        "decode_mode": "generate_total_fallback",
        "git_commit_hash": git_commit_hash(),
        "notes": [
            "Prefill component timings are approximate because hooks synchronize CUDA.",
            "Some component hooks are nested; raw sums should not be treated as exact decomposition.",
            "Generation timing uses Bolmo generate() total time, not pure cached decode.",
            "Patch stats are captured from local_encoder hook output when available.",
        ],
    }

    raw_records: list[dict[str, Any]] = []
    prompts = get_prompt_set(args.prompt_set)
    first_diagnostic_done = False

    try:
        with torch.inference_mode():
            for p in prompts:
                print(f"[info] prompt={p['id']} warmup x{args.num_warmup}")
                for _ in range(args.num_warmup):
                    warmup_pre = run_prefill(model, tokenizer, p["text"], device, state)
                    _ = run_generation(model, tokenizer, p["text"], device, args.max_new_tokens)
                    if not first_diagnostic_done:
                        first_diagnostic_done = True
                        patch = warmup_pre.get("patch_stats", {})
                        print("\n[diagnostic] === HOOK FIRING REPORT (first warmup) ===")
                        print(f"[diagnostic] hooks fired ({len(warmup_pre.get('hooks_fired', []))}): {warmup_pre.get('hooks_fired', [])}")
                        print(f"[diagnostic] component_times: {list(warmup_pre['component_times_ms'].keys())}")
                        print(f"[diagnostic] global_layers: {list(warmup_pre['global_layer_times_ms'].keys())}")
                        print(f"[diagnostic] decoder_layers: {list(warmup_pre['decoder_layer_times_ms'].keys())}")
                        print(f"[diagnostic] captured boundary_mask shape: {warmup_pre.get('captured_boundary_mask_shape')}")
                        print(f"[diagnostic] captured boundary_logprobs shape: {warmup_pre.get('captured_boundary_logprobs_shape')}")
                        print(f"[diagnostic] patch_stats source: {patch.get('patch_stats_source')}")
                        print(f"[diagnostic] num_patches: {patch.get('num_patches')}")
                        print(f"[diagnostic] boundary_conf available: {patch.get('boundary_confidences_available')}")
                        print(f"[diagnostic] boundary_conf source: {patch.get('boundary_confidence_source')}")
                        expected = {"local_encoder", "boundary", "global_backbone", "local_decoder", "output_head"}
                        got = set(warmup_pre["component_times_ms"].keys())
                        missing = expected - got
                        if missing:
                            print(f"[diagnostic] WARNING: components with no timing: {missing}")
                            for m in missing:
                                mod = modules.get(m)
                                print(f"[diagnostic]   {m}: module={type(mod).__name__ if mod else 'None'}, resolved_path={component_paths.get(m)}")
                        else:
                            print("[diagnostic] all 5 components captured timing")
                        prefill_ms = warmup_pre["prefill_time_ms"]
                        unacc = warmup_pre.get("unaccounted_time_ms", 0.0)
                        print(f"[diagnostic] unaccounted_time: {unacc:.1f}ms ({unacc / prefill_ms * 100:.1f}% of prefill)")
                        print("[diagnostic] ===========================================\n")

                print(f"[info] prompt={p['id']} measured repeats x{args.num_repeats}")
                for rep in range(args.num_repeats):
                    pre = run_prefill(model, tokenizer, p["text"], device, state)
                    gen = run_generation(model, tokenizer, p["text"], device, args.max_new_tokens)
                    gb_frac = pre["global_backbone_total_time_ms"] / pre["prefill_time_ms"] if pre.get("global_backbone_total_time_ms") else None
                    dec_frac = pre["local_decoder_total_time_ms"] / pre["prefill_time_ms"] if pre.get("local_decoder_total_time_ms") else None

                    if gb_frac is not None and gb_frac > 1.0:
                        print(f"[warn] global_backbone_fraction_of_prefill={gb_frac:.2f} > 1. Hook timing overhead may be inflating component times.")
                    if dec_frac is not None and dec_frac > 1.0:
                        print(f"[warn] local_decoder_fraction_of_prefill={dec_frac:.2f} > 1. Hook timing overhead may be inflating component times.")

                    rec = {
                        "prompt_id": p["id"],
                        "prompt_name": p["name"],
                        "input_text": p["text"],
                        "repeat": rep,
                        "number_of_repeats": args.num_repeats,
                        "model_id": args.model_id,
                        "device": str(device),
                        "dtype": args.dtype,
                        "global_backbone_fraction_of_prefill": gb_frac,
                        "local_decoder_fraction_of_prefill": dec_frac,
                        **pre,
                        **gen,
                    }
                    raw_records.append(rec)
                    gf = f"{gb_frac:.3f}" if gb_frac is not None else "None"
                    df = f"{dec_frac:.3f}" if dec_frac is not None else "None"
                    patches = pre["patch_stats"].get("num_patches")
                    print(
                        f"[info] prompt={p['id']} rep={rep} "
                        f"prefill={pre['prefill_time_ms']:.1f}ms "
                        f"generate_total={gen['generate_total_time_ms']:.1f}ms "
                        f"patches={patches} "
                        f"global_frac={gf} "
                        f"dec_frac={df} "
                        f"unaccounted={pre.get('unaccounted_time_ms', 0):.1f}ms "
                        f"hooks={len(pre.get('hooks_fired', []))} "
                        f"gen_bytes={gen.get('generated_bytes')} "
                        f"gen_src={gen.get('generation_continuation_source')}"
                    )
    finally:
        remove_hooks()

    save_results(raw_records, out_dir)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[info] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
