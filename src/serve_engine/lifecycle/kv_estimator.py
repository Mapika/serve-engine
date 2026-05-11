from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ACTIVATION_OVERHEAD = 1.15


@dataclass(frozen=True)
class KVEstimateInput:
    model_dir: Path
    max_model_len: int
    target_concurrency: int
    dtype: str


def _dtype_bytes(dtype: str, torch_dtype: str | None) -> int:
    if dtype == "fp8":
        return 1
    if dtype in ("fp16", "bf16"):
        return 2
    if dtype == "auto":
        if torch_dtype in ("float16", "bfloat16"):
            return 2
        if torch_dtype == "float32":
            return 4
        return 2  # default
    return 2


def read_model_config(model_dir: Path) -> dict:
    p = Path(model_dir) / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"no config.json under {model_dir}")
    return json.loads(p.read_text())


def _estimate_param_bytes(cfg: dict, dtype_bytes: int) -> int:
    """Rough parameter count from config; used when safetensors metadata unavailable."""
    L = int(cfg.get("num_hidden_layers", 0))
    H = int(cfg.get("hidden_size", 0))
    vocab = int(cfg.get("vocab_size", 0))
    if L == 0 or H == 0:
        return 0
    backbone = 12 * L * H * H
    embed = vocab * H * 2  # input + output embeddings
    return (backbone + embed) * dtype_bytes


def estimate_vram_mb(inp: KVEstimateInput) -> int:
    cfg = read_model_config(inp.model_dir)
    torch_dtype = cfg.get("torch_dtype")
    dtype_bytes = _dtype_bytes(inp.dtype, torch_dtype)

    n_layers = int(cfg.get("num_hidden_layers", 0))
    hidden = int(cfg.get("hidden_size", 0))
    n_heads = int(cfg.get("num_attention_heads", 1))
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", hidden // n_heads if n_heads else 0))

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
    kv_bytes = kv_bytes_per_token * inp.max_model_len * inp.target_concurrency
    weights_bytes = _estimate_param_bytes(cfg, dtype_bytes)

    total = (weights_bytes + kv_bytes) * ACTIVATION_OVERHEAD
    return math.ceil(total / 1024 / 1024)
