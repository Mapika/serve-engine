from unittest.mock import MagicMock

import pytest

from serve_engine.lifecycle.docker_client import ContainerHandle, DockerClient


@pytest.fixture
def fake_docker():
    """A MagicMock that imitates the bits of docker.from_env() we touch."""
    client = MagicMock()
    container = MagicMock()
    container.id = "abc123"
    container.name = "vllm-llama-1b"
    container.attrs = {
        "NetworkSettings": {"Networks": {"serve-engines": {"IPAddress": "172.20.0.5"}}}
    }
    client.containers.run.return_value = container
    return client


def test_run_container_returns_handle(fake_docker):
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    handle = dc.run(
        image="vllm/vllm-openai:v0.7.3",
        name="vllm-llama-1b",
        command=["--model", "/models/x"],
        environment={},
        kwargs={"ipc_mode": "host"},
        volumes={"/host/models": {"bind": "/models", "mode": "ro"}},
        internal_port=8000,
    )
    assert isinstance(handle, ContainerHandle)
    assert handle.id == "abc123"
    assert handle.address == "vllm-llama-1b"
    assert handle.port == 8000


def test_run_creates_network_if_missing(fake_docker):
    from docker.errors import NotFound
    fake_docker.networks.get.side_effect = NotFound("not found")
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.ensure_network()
    fake_docker.networks.create.assert_called_once_with("serve-engines", driver="bridge")


def test_ensure_network_propagates_other_errors(fake_docker):
    fake_docker.networks.get.side_effect = RuntimeError("daemon connection refused")
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    with pytest.raises(RuntimeError, match="daemon connection refused"):
        dc.ensure_network()


def test_run_skips_network_create_if_present(fake_docker):
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.ensure_network()
    fake_docker.networks.create.assert_not_called()


def test_stop_calls_remove(fake_docker):
    container = MagicMock()
    fake_docker.containers.get.return_value = container
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.stop("abc123", timeout=10)
    container.stop.assert_called_once_with(timeout=10)
    container.remove.assert_called_once()


def test_stop_is_idempotent_for_missing_container(fake_docker):
    from docker.errors import NotFound
    fake_docker.containers.get.side_effect = NotFound("gone")
    dc = DockerClient(client=fake_docker, network_name="serve-engines")
    dc.stop("abc123", timeout=10)  # must not raise
