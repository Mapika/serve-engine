
import pytest

from serve_engine.store import db
from serve_engine.store import models as model_store


def test_init_schema_creates_tables(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.init_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r[0] for r in rows}
    assert "models" in table_names
    assert "deployments" in table_names
    assert "_migrations" in table_names


def test_init_schema_is_idempotent(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.init_schema(conn)
    db.init_schema(conn)  # second call must not error

    applied = conn.execute(
        "SELECT COUNT(*) FROM _migrations WHERE filename='001_initial.sql'"
    ).fetchone()[0]
    assert applied == 1


def _fresh(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.init_schema(conn)
    return conn


def test_add_and_get_model(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="llama-1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct")
    assert m.id is not None
    assert m.name == "llama-1b"
    assert m.revision == "main"

    fetched = model_store.get_by_name(conn, "llama-1b")
    assert fetched is not None
    assert fetched.id == m.id


def test_add_duplicate_model_raises(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="x", hf_repo="org/x")
    with pytest.raises(model_store.AlreadyExists):
        model_store.add(conn, name="x", hf_repo="org/x")


def test_list_models_empty(tmp_path):
    conn = _fresh(tmp_path)
    assert model_store.list_all(conn) == []


def test_list_models_returns_in_creation_order(tmp_path):
    conn = _fresh(tmp_path)
    model_store.add(conn, name="a", hf_repo="org/a")
    model_store.add(conn, name="b", hf_repo="org/b")
    rows = model_store.list_all(conn)
    assert [m.name for m in rows] == ["a", "b"]


def test_set_local_path(tmp_path):
    conn = _fresh(tmp_path)
    m = model_store.add(conn, name="x", hf_repo="org/x")
    model_store.set_local_path(conn, m.id, "/var/x")
    fetched = model_store.get_by_name(conn, "x")
    assert fetched.local_path == "/var/x"
