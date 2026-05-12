from unittest.mock import patch

from serve_engine.lifecycle.adapter_downloader import download_adapter


def test_download_adapter_calls_snapshot_download(tmp_path):
    fake_dir = tmp_path / "snap"
    fake_dir.mkdir()
    # Two files totaling 3 MB (3 * 1024 * 1024 + 1 byte to force ceil)
    (fake_dir / "adapter_model.safetensors").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    (fake_dir / "adapter_config.json").write_bytes(b"x" * (1 * 1024 * 1024))

    with patch(
        "serve_engine.lifecycle.adapter_downloader.snapshot_download",
        return_value=str(fake_dir),
    ) as mock_sd:
        path, size_mb = download_adapter(
            hf_repo="org/some-lora",
            revision="main",
            cache_dir=tmp_path,
        )
    assert path == str(fake_dir)
    assert size_mb == 4  # 3 MB + 1 byte rounds up to 4
    kwargs = mock_sd.call_args.kwargs
    assert kwargs["repo_id"] == "org/some-lora"
    assert kwargs["revision"] == "main"
    assert kwargs["cache_dir"] == str(tmp_path)


def test_download_adapter_size_zero_for_empty_dir(tmp_path):
    fake_dir = tmp_path / "empty_snap"
    fake_dir.mkdir()
    with patch(
        "serve_engine.lifecycle.adapter_downloader.snapshot_download",
        return_value=str(fake_dir),
    ):
        _, size_mb = download_adapter(
            hf_repo="o/x", revision="main", cache_dir=tmp_path,
        )
    assert size_mb == 0


def test_download_adapter_recurses_into_subdirs(tmp_path):
    fake_dir = tmp_path / "snap"
    fake_dir.mkdir()
    nested = fake_dir / "subdir" / "deeper"
    nested.mkdir(parents=True)
    (nested / "weights.bin").write_bytes(b"x" * (5 * 1024 * 1024))
    with patch(
        "serve_engine.lifecycle.adapter_downloader.snapshot_download",
        return_value=str(fake_dir),
    ):
        _, size_mb = download_adapter(
            hf_repo="o/x", revision="main", cache_dir=tmp_path,
        )
    assert size_mb == 5
