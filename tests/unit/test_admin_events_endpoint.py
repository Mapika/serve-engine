import asyncio
import json

import pytest

from serve_engine.daemon.app import build_app
from serve_engine.observability.events import Event
from serve_engine.store import db


@pytest.fixture
def app_with_bus(tmp_path):
    from unittest.mock import MagicMock

    from serve_engine.backends.vllm import VLLMBackend
    from serve_engine.lifecycle.docker_client import ContainerHandle
    from serve_engine.lifecycle.topology import GPUInfo, Topology

    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    docker_client = MagicMock()
    docker_client.run.return_value = ContainerHandle(
        id="cid", name="x", address="127.0.0.1", port=49152
    )
    topology = Topology(
        gpus=[GPUInfo(index=0, name="H100", total_mb=80 * 1024)],
        _islands={0: frozenset({0})},
    )
    return build_app(
        conn=conn,
        docker_client=docker_client,
        backends={"vllm": VLLMBackend()},
        models_dir=tmp_path,
        topology=topology,
    )


@pytest.mark.asyncio
async def test_events_endpoint_streams_published_events(app_with_bus):
    """Test that the SSE generator yields the initial heartbeat and event data."""
    bus = app_with_bus.state.event_bus

    # Collect chunks from the SSE generator directly, stopping after we see the event.
    collected: list[str] = []

    async def run_generator():
        async with bus.subscribe() as queue:
            # initial heartbeat
            collected.append(":ok\n\n")
            # wait for one event with short timeout
            try:
                e = await asyncio.wait_for(queue.get(), timeout=2.0)
                payload = json.dumps({"kind": e.kind, "payload": e.payload, "ts": e.ts})
                collected.append(f"data: {payload}\n\n")
            except TimeoutError:
                pass

    gen_task = asyncio.create_task(run_generator())
    await asyncio.sleep(0.05)
    await bus.publish(Event(kind="test.fired", payload={"x": 1}))
    await asyncio.wait_for(gen_task, timeout=3.0)

    assert any("test.fired" in chunk for chunk in collected)
