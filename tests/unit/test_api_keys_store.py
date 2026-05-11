from serve_engine.store import api_keys, db


def _fresh(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def test_create_returns_full_key_once(tmp_path):
    conn = _fresh(tmp_path)
    secret, key = api_keys.create(conn, name="alice", tier="standard")
    assert secret.startswith("sk-")
    assert len(secret) >= 32
    assert key.prefix == secret[:12]
    assert key.name == "alice"
    assert key.tier == "standard"


def test_verify_matches_hashed(tmp_path):
    conn = _fresh(tmp_path)
    secret, _ = api_keys.create(conn, name="alice", tier="standard")
    found = api_keys.verify(conn, secret)
    assert found is not None
    assert found.name == "alice"


def test_verify_rejects_wrong_secret(tmp_path):
    conn = _fresh(tmp_path)
    api_keys.create(conn, name="alice", tier="standard")
    assert api_keys.verify(conn, "sk-wrong") is None


def test_verify_rejects_revoked(tmp_path):
    conn = _fresh(tmp_path)
    secret, key = api_keys.create(conn, name="alice", tier="standard")
    api_keys.revoke(conn, key.id)
    assert api_keys.verify(conn, secret) is None


def test_list_all_excludes_secret(tmp_path):
    conn = _fresh(tmp_path)
    api_keys.create(conn, name="a", tier="standard")
    api_keys.create(conn, name="b", tier="standard")
    rows = api_keys.list_all(conn)
    assert len(rows) == 2
    assert all(not hasattr(r, "secret") for r in rows)


def test_per_key_overrides(tmp_path):
    conn = _fresh(tmp_path)
    _, key = api_keys.create(
        conn, name="alice", tier="standard", rpm_override=120, tpm_override=50_000,
    )
    fetched = api_keys.get_by_id(conn, key.id)
    assert fetched.rpm_override == 120
    assert fetched.tpm_override == 50_000
    assert fetched.rpd_override is None


def test_count_active(tmp_path):
    conn = _fresh(tmp_path)
    assert api_keys.count_active(conn) == 0
    _, k1 = api_keys.create(conn, name="a", tier="standard")
    api_keys.create(conn, name="b", tier="standard")
    assert api_keys.count_active(conn) == 2
    api_keys.revoke(conn, k1.id)
    assert api_keys.count_active(conn) == 1
