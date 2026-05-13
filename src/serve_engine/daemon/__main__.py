from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import structlog
import uvicorn

from serve_engine import config
from serve_engine.backends.sglang import SGLangBackend
from serve_engine.backends.trtllm import TRTLLMBackend
from serve_engine.backends.vllm import VLLMBackend
from serve_engine.daemon.app import build_apps
from serve_engine.lifecycle.docker_client import DockerClient
from serve_engine.store import db


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


async def serve(public_host: str, public_port: int, sock_path: Path) -> None:
    config.SERVE_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    conn = db.connect(config.DB_PATH)
    db.init_schema(conn)

    docker_client = DockerClient(network_name=config.DOCKER_NETWORK_NAME)
    docker_client.ensure_network()

    from serve_engine.lifecycle.topology import read_topology
    topology = read_topology()
    log_ = logging.getLogger(__name__)
    log_.info(
        "topology: %d GPUs, islands=%s",
        len(topology.gpus),
        [list(topology.nvlink_island(g.index)) for g in topology.gpus],
    )

    from serve_engine.backends.manifest import load_manifest
    manifest = load_manifest()
    backends = {
        "vllm": VLLMBackend(manifest["vllm"]),
        "sglang": SGLangBackend(manifest["sglang"]),
        "trtllm": TRTLLMBackend(manifest["trtllm"]),
    }
    tcp_app, uds_app = build_apps(
        conn=conn,
        docker_client=docker_client,
        backends=backends,
        models_dir=config.MODELS_DIR,
        topology=topology,
        configs_dir=config.CONFIGS_DIR,
        snapshots_dir=config.SNAPSHOTS_DIR,
    )

    if sock_path.exists():
        sock_path.unlink()

    tcp_cfg = uvicorn.Config(app=tcp_app, host=public_host, port=public_port, log_level="info")
    uds_cfg = uvicorn.Config(app=uds_app, uds=str(sock_path), log_level="info")
    tcp_server = uvicorn.Server(tcp_cfg)
    uds_server = uvicorn.Server(uds_cfg)
    await asyncio.gather(tcp_server.serve(), uds_server.serve())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="serve-engine-daemon")
    p.add_argument("--host", default=config.DEFAULT_PUBLIC_HOST)
    p.add_argument("--port", default=config.DEFAULT_PUBLIC_PORT, type=int)
    p.add_argument("--sock", default=str(config.SOCK_PATH))
    args = p.parse_args(argv)

    configure_logging()
    asyncio.run(serve(args.host, args.port, Path(args.sock)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
