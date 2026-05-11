from __future__ import annotations

import logging
from dataclasses import dataclass

import docker  # type: ignore[import-untyped]
from docker.errors import NotFound  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContainerHandle:
    id: str
    name: str
    address: str       # host or container name reachable from daemon
    port: int          # port on `address` to talk HTTP to


class DockerClient:
    def __init__(self, *, client: object | None = None, network_name: str):
        self._client = client or docker.from_env()
        self._network_name = network_name

    def ensure_network(self) -> None:
        try:
            self._client.networks.get(self._network_name)
        except Exception:
            log.info("creating docker network %s", self._network_name)
            self._client.networks.create(self._network_name, driver="bridge")

    def run(
        self,
        *,
        image: str,
        name: str,
        command: list[str],
        environment: dict[str, str],
        kwargs: dict[str, object],
        volumes: dict[str, dict[str, str]],
        internal_port: int,
    ) -> ContainerHandle:
        container = self._client.containers.run(
            image=image,
            command=command,
            name=name,
            environment=environment,
            volumes=volumes,
            network=self._network_name,
            detach=True,
            **kwargs,
        )
        return ContainerHandle(
            id=container.id,
            name=name,
            address=name,  # talk by container name on the bridge
            port=internal_port,
        )

    def stop(self, container_id: str, *, timeout: int = 30) -> None:
        try:
            c = self._client.containers.get(container_id)
        except NotFound:
            return
        c.stop(timeout=timeout)
        c.remove()

    def stream_logs(self, container_id: str, *, follow: bool = False):
        c = self._client.containers.get(container_id)
        return c.logs(stream=True, follow=follow)

    def pull(self, image: str) -> None:
        self._client.images.pull(image)
