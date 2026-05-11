
from serve_engine.store import db


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
