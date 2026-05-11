from pathlib import Path

from serve_engine.backends.selection import (
    load_selection,
    pick_backend,
)


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "selection.yaml"
    p.write_text(content)
    return p


def test_default_picks_vllm(tmp_path):
    cfg = load_selection(_write(tmp_path, "rules: []\ndefault: vllm\n"))
    assert pick_backend(cfg, "Llama-3.1-70B-Instruct") == "vllm"


def test_pattern_match(tmp_path):
    cfg = load_selection(_write(
        tmp_path,
        "rules:\n"
        "  - pattern: '*deepseek-v3*'\n"
        "    backend: sglang\n"
        "default: vllm\n",
    ))
    assert pick_backend(cfg, "deepseek-v3-671b") == "sglang"
    assert pick_backend(cfg, "Llama-3.1-70B") == "vllm"


def test_first_match_wins(tmp_path):
    cfg = load_selection(_write(
        tmp_path,
        "rules:\n"
        "  - pattern: '*qwen*'\n"
        "    backend: sglang\n"
        "  - pattern: '*qwen*-vl*'\n"
        "    backend: vllm\n"
        "default: vllm\n",
    ))
    # First rule matches qwen-2.5-vl-7b too; first match wins
    assert pick_backend(cfg, "qwen-2.5-vl-7b") == "sglang"


def test_case_insensitive(tmp_path):
    cfg = load_selection(_write(
        tmp_path,
        "rules:\n"
        "  - pattern: '*DEEPSEEK*'\n"
        "    backend: sglang\n"
        "default: vllm\n",
    ))
    assert pick_backend(cfg, "deepseek-v3") == "sglang"


def test_load_default_path_uses_package_resource():
    cfg = load_selection()
    assert cfg.default in ("vllm", "sglang")
    assert isinstance(cfg.rules, list)
