from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_adapter(
    *,
    hf_repo: str,
    revision: str,
    cache_dir: Path,
) -> tuple[str, int]:
    """Download a LoRA / DoRA adapter via HF Hub. Returns (local_path, size_mb).

    Adapter dirs are typically <500 MB so this stays synchronous +
    simple, matching `lifecycle.downloader.download_model`. The size
    return value is stored in `adapters.size_mb` for UI display and
    quota planning; computed by walking the snapshot dir.
    """
    local_path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    size_bytes = sum(
        f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()
    )
    size_mb = (size_bytes + 1024 * 1024 - 1) // (1024 * 1024)  # ceil
    return local_path, int(size_mb)
