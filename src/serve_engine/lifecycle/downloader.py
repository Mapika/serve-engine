from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download

ProgressFn = Callable[[str], None]


def download_model(
    *,
    hf_repo: str,
    revision: str,
    cache_dir: Path,
    on_event: ProgressFn | None = None,
) -> str:
    def emit(msg: str) -> None:
        if on_event is not None:
            on_event(msg)

    emit(f"download started: {hf_repo}@{revision}")
    path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    emit(f"download complete: {path}")
    return path
