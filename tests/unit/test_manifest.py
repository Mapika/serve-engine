import pytest

from serve_engine.backends.manifest import Headroom, load_manifest


def test_load_default_manifest_has_vllm_and_sglang():
    m = load_manifest()
    assert "vllm" in m and "sglang" in m
    assert m["vllm"].image_default.startswith("vllm/vllm-openai:")
    assert m["sglang"].image_default.startswith("lmsysorg/sglang:")
    assert m["vllm"].internal_port == 8000
    assert m["sglang"].internal_port == 30000


def test_headroom_effective_util_uses_max_of_three():
    hr = Headroom(factor=1.5, min_extra_mb=2048, min_floor_pct=15)
    # Small model: floor wins
    util = hr.effective_util(reserved_mb=1000, per_gpu_mb=80000)
    assert util == pytest.approx(0.15)
    # Mid model: factor or +extra wins
    util = hr.effective_util(reserved_mb=40000, per_gpu_mb=80000)
    # max(60000, 42048, 12000) = 60000 → 0.75
    assert util == pytest.approx(0.75)
    # Large model: factor still wins but clamped to 0.95
    util = hr.effective_util(reserved_mb=70000, per_gpu_mb=80000)
    assert util == 0.95


def test_headroom_clamps_to_min():
    hr = Headroom(factor=1.0, min_extra_mb=0, min_floor_pct=0)
    util = hr.effective_util(reserved_mb=100, per_gpu_mb=80000)
    assert util == 0.05  # min floor
