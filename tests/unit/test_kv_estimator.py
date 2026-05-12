import json

import pytest

from serve_engine.lifecycle.kv_estimator import (
    KVEstimateInput,
    estimate_vram_mb,
    read_model_config,
)


def _write_config(tmp_path, **overrides):
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "torch_dtype": "bfloat16",
        "vocab_size": 32000,
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return tmp_path


def test_estimate_basic(tmp_path):
    model_dir = _write_config(tmp_path)
    inp = KVEstimateInput(
        model_dir=model_dir,
        max_model_len=4096,
        target_concurrency=8,
        dtype="auto",
    )
    mb = estimate_vram_mb(inp)
    assert mb > 0
    assert 1000 < mb < 10000


def test_estimate_handles_gqa(tmp_path):
    full = _write_config(tmp_path)
    full_mb = estimate_vram_mb(
        KVEstimateInput(model_dir=full, max_model_len=4096,
                        target_concurrency=8, dtype="bf16")
    )

    gqa_dir = tmp_path / "gqa"
    gqa_dir.mkdir()
    _write_config(gqa_dir, num_key_value_heads=4)
    gqa_mb = estimate_vram_mb(
        KVEstimateInput(model_dir=gqa_dir, max_model_len=4096,
                        target_concurrency=8, dtype="bf16")
    )
    assert gqa_mb < full_mb


def test_dtype_fp8_halves_kv(tmp_path):
    md = _write_config(tmp_path)
    bf16 = estimate_vram_mb(KVEstimateInput(
        model_dir=md, max_model_len=4096, target_concurrency=8, dtype="bf16"))
    fp8 = estimate_vram_mb(KVEstimateInput(
        model_dir=md, max_model_len=4096, target_concurrency=8, dtype="fp8"))
    assert fp8 < bf16


def test_read_model_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_model_config(tmp_path)


def test_moe_config_dwarfs_dense(tmp_path):
    """An MoE FFN (N experts x moe_intermediate_size) should dwarf the dense FFN.

    Pre-fix the estimator ignored num_experts/moe_intermediate_size and produced
    the same number for both, which is what made the headroom calc undershoot
    by ~10x on Qwen3.6-35B-A3B-FP8.
    """
    dense = _write_config(tmp_path, hidden_size=2048, intermediate_size=4096)
    moe_dir = tmp_path / "moe"
    moe_dir.mkdir()
    _write_config(
        moe_dir,
        hidden_size=2048,
        intermediate_size=4096,
        num_experts=256,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )
    inp = lambda d: KVEstimateInput(  # noqa: E731
        model_dir=d, max_model_len=4096, target_concurrency=8, dtype="bf16",
    )
    moe_mb = estimate_vram_mb(inp(moe_dir))
    dense_mb = estimate_vram_mb(inp(dense))
    assert moe_mb > dense_mb * 5, (moe_mb, dense_mb)


def test_hybrid_layer_types_reduce_kv(tmp_path):
    """layer_types with linear_attention layers should shrink the KV estimate."""
    base = dict(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        torch_dtype="bfloat16",
        vocab_size=152000,
    )
    all_attn = _write_config(tmp_path, **base)
    hybrid_dir = tmp_path / "hybrid"
    hybrid_dir.mkdir()
    layer_types = ["linear_attention"] * 30 + ["full_attention"] * 10
    _write_config(hybrid_dir, layer_types=layer_types, **base)

    inp = lambda d: KVEstimateInput(  # noqa: E731
        model_dir=d, max_model_len=65536, target_concurrency=8, dtype="bf16",
    )
    full_mb = estimate_vram_mb(inp(all_attn))
    hybrid_mb = estimate_vram_mb(inp(hybrid_dir))
    # KV is ~4x smaller (10 attn layers vs 40), so total should be meaningfully
    # smaller at the long-context regime where KV dominates.
    assert hybrid_mb < full_mb, (hybrid_mb, full_mb)


def test_quantization_config_fp8_halves_weights(tmp_path):
    """A checkpoint with quantization_config.quant_method='fp8' weighs ~half."""
    bf16 = _write_config(tmp_path)
    fp8_dir = tmp_path / "fp8"
    fp8_dir.mkdir()
    _write_config(fp8_dir, quantization_config={"quant_method": "fp8"})
    inp = lambda d: KVEstimateInput(  # noqa: E731
        model_dir=d, max_model_len=4096, target_concurrency=8, dtype="auto",
    )
    bf16_mb = estimate_vram_mb(inp(bf16))
    fp8_mb = estimate_vram_mb(inp(fp8_dir))
    assert fp8_mb < bf16_mb, (fp8_mb, bf16_mb)


def test_text_config_nested_arch_fields(tmp_path):
    """Multimodal configs put arch fields under text_config; we should find them."""
    cfg = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "torch_dtype": "bfloat16",
            "vocab_size": 152000,
            "num_experts": 256,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "layer_types": ["linear_attention"] * 30 + ["full_attention"] * 10,
        },
        "quantization_config": {"quant_method": "fp8"},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    mb = estimate_vram_mb(KVEstimateInput(
        model_dir=tmp_path, max_model_len=65536, target_concurrency=8, dtype="auto",
    ))
    # Qwen3.6-35B-A3B at FP8 on-disk is ~35 GB. Estimate should be in the
    # right order of magnitude — i.e. tens of GB, not hundreds of MB.
    assert 25_000 < mb < 80_000, mb
