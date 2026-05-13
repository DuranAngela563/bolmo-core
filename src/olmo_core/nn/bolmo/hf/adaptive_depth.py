"""
adaptive_depth.py — Backbone depth strategies for Bolmo
========================================================

Design
------
modeling_bolmo.py exposes two things for adaptive depth:

    1. BolmoModel.global_backbone_strategy   (None = default path)
    2. BolmoModel._run_global_layers(...)     (canonical layer loop)

All experiment logic lives here.  Strategies are thin functions that
call model._run_global_layers with the right start/end slice.
No code in this file duplicates the per-layer invocation logic.

Usage
-----
    from modeling_bolmo import BolmoForCausalLM
    from adaptive_depth import apply_depth_strategy, clear_depth_strategy

    model = BolmoForCausalLM.from_pretrained(...)

    # Phase 1 — fixed truncation at 50 % of layers
    apply_depth_strategy(model.model, "fixed", depth_fraction=0.5)

    # Back to the original unmodified path (true baseline)
    clear_depth_strategy(model.model)

    # Sanity-check: strategy that runs all layers (must match original)
    apply_depth_strategy(model.model, "full")
"""

from __future__ import annotations

from typing import Callable, Any

import torch
import torch.nn as nn


# ======================================================================
# Type alias
# ======================================================================
# Every strategy function has this signature:
#   (bolmo_model, h_patch, causal_mask_mapping, position_ids,
#    past_key_values, cache_position, position_embeddings_mapping,
#    **kwargs)  ->  torch.Tensor

BackboneFn = Callable[..., torch.Tensor]


# ======================================================================
# Registry
# ======================================================================
_STRATEGIES: dict[str, Callable[..., BackboneFn]] = {}


def register_strategy(name: str):
    """Decorator.  The decorated callable must return a BackboneFn."""
    def _decorator(factory):
        _STRATEGIES[name] = factory
        return factory
    return _decorator


def list_strategies() -> list[str]:
    return sorted(_STRATEGIES.keys())


# ======================================================================
# Strategies
# ======================================================================

@register_strategy("full")
def make_full_strategy(**kwargs) -> BackboneFn:
    """Run all global backbone layers via the model's canonical loop.

    Use this as a sanity check: its output must be bit-identical to
    the original (no-strategy) path.
    """
    def run(bolmo_model, h_patch, causal_mask_mapping, position_ids,
            past_key_values, cache_position, position_embeddings_mapping,
            **kw):
        return bolmo_model._run_global_layers(
            h_patch, causal_mask_mapping, position_ids,
            past_key_values, cache_position, position_embeddings_mapping,
            start_layer=0,
            end_layer=bolmo_model.config.num_hidden_layers,
            **kw,
        )
    return run


@register_strategy("fixed")
def make_fixed_strategy(
    depth_fraction: float | None = None,
    num_layers: int | None = None,
    _total_layers: int | None = None,
    **kwargs,
) -> BackboneFn:
    """Phase 1 — run only the first K global backbone layers.

    Specify EITHER depth_fraction OR num_layers (not both).

    Validation is strict:
        depth_fraction  must be in (0, 1]
        num_layers      must be in [1, total_layers]

    The resolved layer count is baked into the returned function so
    validation happens once at strategy-creation time, not per forward.

    _total_layers is injected by apply_depth_strategy(); do not set manually.
    """
    if depth_fraction is not None and num_layers is not None:
        raise ValueError("Specify depth_fraction or num_layers, not both.")
    if depth_fraction is None and num_layers is None:
        raise ValueError("Specify either depth_fraction or num_layers.")

    if num_layers is not None:
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if _total_layers is not None and num_layers > _total_layers:
            raise ValueError(
                f"num_layers={num_layers} exceeds total layers={_total_layers}"
            )
        resolved_k = num_layers
    else:
        assert depth_fraction is not None
        if not (0 < depth_fraction <= 1.0):
            raise ValueError(
                f"depth_fraction must be in (0, 1], got {depth_fraction}"
            )
        if _total_layers is None:
            raise ValueError(
                "Cannot resolve depth_fraction without _total_layers. "
                "Use apply_depth_strategy() which injects this automatically."
            )
        resolved_k = max(1, round(_total_layers * depth_fraction))

    def run(bolmo_model, h_patch, causal_mask_mapping, position_ids,
            past_key_values, cache_position, position_embeddings_mapping,
            **kw):
        return bolmo_model._run_global_layers(
            h_patch, causal_mask_mapping, position_ids,
            past_key_values, cache_position, position_embeddings_mapping,
            start_layer=0,
            end_layer=resolved_k,
            **kw,
        )

    run.resolved_num_layers = resolved_k
    return run


# Future strategies added here — see thesis notes for design considerations.


# ======================================================================
# Public API
# ======================================================================

def apply_depth_strategy(
    bolmo_model: nn.Module,
    strategy: str,
    **strategy_kwargs,
) -> None:
    """Attach a depth strategy to a BolmoModel instance.

    After this call, every forward pass uses the chosen strategy for the
    global backbone loop.  The rest of the model (local encoder, local
    decoder, boundary predictor) is completely untouched.

    Parameters
    ----------
    bolmo_model : BolmoModel
        The ``.model`` attribute of a BolmoForCausalLM.
    strategy : str
        One of the registered names (see ``list_strategies()``).
    **strategy_kwargs
        Forwarded to the strategy factory (e.g. ``depth_fraction=0.5``).
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {list_strategies()}"
        )

    # Inject total_layers so strategies can validate at creation time.
    strategy_kwargs["_total_layers"] = bolmo_model.config.num_hidden_layers

    backbone_fn = _STRATEGIES[strategy](**strategy_kwargs)
    bolmo_model.global_backbone_strategy = backbone_fn

    # Metadata for logging / experiment tracking
    bolmo_model._depth_strategy_name = strategy
    bolmo_model._depth_strategy_kwargs = {
        k: v for k, v in strategy_kwargs.items() if k != "_total_layers"
    }


def clear_depth_strategy(bolmo_model: nn.Module) -> None:
    """Remove any attached strategy, restoring the original default path.

    Use this to obtain the true unmodified baseline.
    """
    bolmo_model.global_backbone_strategy = None
    bolmo_model._depth_strategy_name = "original"
    bolmo_model._depth_strategy_kwargs = {}


def get_active_strategy(bolmo_model: nn.Module) -> dict:
    """Return a dict describing the currently active strategy."""
    return {
        "strategy": getattr(bolmo_model, "_depth_strategy_name", "original"),
        "kwargs": getattr(bolmo_model, "_depth_strategy_kwargs", {}),
    }


def get_effective_num_layers(bolmo_model: nn.Module) -> int:
    """Return how many backbone layers will actually execute."""
    total = bolmo_model.config.num_hidden_layers
    strategy_fn = bolmo_model.global_backbone_strategy
    if strategy_fn is not None and hasattr(strategy_fn, "resolved_num_layers"):
        return strategy_fn.resolved_num_layers
    return total
