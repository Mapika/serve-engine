"""Pure-resolution tests for adapter_router. The async ensure_adapter_loaded
helper is exercised in tests/unit/test_admin_adapter_endpoints.py and
tests/unit/test_proxy_adapter_dispatch.py."""
import pytest

from serve_engine.lifecycle.adapter_router import (
    UnknownModel,
    find_deployment_for,
    resolve_target,
)
from serve_engine.store import adapters as ad_store
from serve_engine.store import db
from serve_engine.store import deployment_adapters as da_store
from serve_engine.store import deployments as dep_store
from serve_engine.store import models as model_store


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


# ---- resolve_target ----

def test_resolve_target_bare_base_unchanged(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    t = resolve_target(conn, "qwen3-7b")
    assert t.base_model_name == "qwen3-7b"
    assert t.adapter_name is None


def test_resolve_target_bare_adapter_returns_base_and_adapter(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    ad_store.add(
        conn, name="tone-formal", base_model_name="qwen3-7b", hf_repo="o/lora",
    )
    t = resolve_target(conn, "tone-formal")
    assert t.base_model_name == "qwen3-7b"
    assert t.adapter_name == "tone-formal"


def test_resolve_target_composite_form(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    ad_store.add(
        conn, name="tone-formal", base_model_name="qwen3-7b", hf_repo="o/lora",
    )
    t = resolve_target(conn, "qwen3-7b:tone-formal")
    assert t.base_model_name == "qwen3-7b"
    assert t.adapter_name == "tone-formal"


def test_resolve_target_composite_with_wrong_base_rejected(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    model_store.add(conn, name="llama-8b", hf_repo="meta/L8B")
    ad_store.add(
        conn, name="tone-formal", base_model_name="qwen3-7b", hf_repo="o/lora",
    )
    with pytest.raises(UnknownModel, match="belongs to base"):
        resolve_target(conn, "llama-8b:tone-formal")


def test_resolve_target_composite_unknown_adapter_rejected(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    with pytest.raises(UnknownModel):
        resolve_target(conn, "qwen3-7b:nope")


def test_resolve_target_unknown_bare_falls_through_as_base(tmp_path):
    """Bare names that don't match an adapter return as a base candidate;
    find_deployment_for handles 'no such base' via returning None."""
    conn = _fresh(tmp_path)
    t = resolve_target(conn, "totally-unknown")
    assert t.base_model_name == "totally-unknown"
    assert t.adapter_name is None


# ---- find_deployment_for ----

def _seed_dep(conn, *, model_id: int, max_loras: int = 0, status: str = "ready"):
    d = dep_store.create(
        conn, model_id=model_id, backend="vllm", image_tag="vllm:test",
        gpu_ids=[0], tensor_parallel=1, max_model_len=4096, dtype="auto",
        max_loras=max_loras,
    )
    dep_store.set_container(
        conn, d.id, container_id=f"c{d.id}", container_name=f"x{d.id}",
        container_port=49152 + d.id, container_address="127.0.0.1",
    )
    dep_store.update_status(conn, d.id, status)
    return d


def test_find_deployment_for_bare_base(tmp_path):
    conn = _fresh(tmp_path)
    base = model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    d = _seed_dep(conn, model_id=base.id)
    found = find_deployment_for(conn, "qwen3-7b", None)
    assert found is not None
    assert found.id == d.id


def test_find_deployment_for_adapter_prefers_already_loaded(tmp_path):
    conn = _fresh(tmp_path)
    base = model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    a = ad_store.add(
        conn, name="x", base_model_name="qwen3-7b", hf_repo="o/lora",
    )
    _seed_dep(conn, model_id=base.id, max_loras=4)  # candidate without adapter
    d_with_adapter = _seed_dep(conn, model_id=base.id, max_loras=4)
    da_store.attach(conn, d_with_adapter.id, a.id)
    found = find_deployment_for(conn, "qwen3-7b", "x")
    assert found is not None
    assert found.id == d_with_adapter.id


def test_find_deployment_for_adapter_with_free_slot_picks_freer_over_full(tmp_path):
    """Between deployment with free slots and deployment that would need
    eviction, prefer the one with free slots."""
    conn = _fresh(tmp_path)
    base = model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    other_a = ad_store.add(
        conn, name="other", base_model_name="qwen3-7b", hf_repo="o/o",
    )
    target_a = ad_store.add(
        conn, name="target", base_model_name="qwen3-7b", hf_repo="o/t",
    )
    d_full = _seed_dep(conn, model_id=base.id, max_loras=1)
    da_store.attach(conn, d_full.id, other_a.id)  # slot now full
    d_free = _seed_dep(conn, model_id=base.id, max_loras=4)
    found = find_deployment_for(conn, "qwen3-7b", target_a.name)
    assert found is not None
    assert found.id == d_free.id


def test_find_deployment_for_adapter_skips_lora_disabled_deployment(tmp_path):
    conn = _fresh(tmp_path)
    base = model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    ad_store.add(conn, name="x", base_model_name="qwen3-7b", hf_repo="o/lora")
    _seed_dep(conn, model_id=base.id, max_loras=0)  # no LoRA
    found = find_deployment_for(conn, "qwen3-7b", "x")
    assert found is None


def test_find_deployment_for_unknown_adapter_returns_none(tmp_path):
    conn = _fresh(tmp_path)
    base = model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    _seed_dep(conn, model_id=base.id, max_loras=4)
    found = find_deployment_for(conn, "qwen3-7b", "no-such-adapter")
    assert found is None


def test_find_deployment_for_no_ready_returns_none(tmp_path):
    conn = _fresh(tmp_path)
    base = model_store.add(conn, name="qwen3-7b", hf_repo="o/qwen")
    _seed_dep(conn, model_id=base.id, max_loras=4, status="stopped")
    ad_store.add(conn, name="x", base_model_name="qwen3-7b", hf_repo="o/lora")
    found = find_deployment_for(conn, "qwen3-7b", "x")
    assert found is None
