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
