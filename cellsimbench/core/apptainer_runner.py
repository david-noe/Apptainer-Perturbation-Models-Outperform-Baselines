"""Apptainer container execution for CellSimBench framework."""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


class ApptainerRunner:
    """Runs model containers via Apptainer instead of Docker.

    Drop-in replacement for DockerRunner: accepts the same run_container()
    arguments and translates them to an `apptainer run` invocation.

    Volume mounts, GPU assignment, and environment variables are all
    preserved. Resource limits (memory, CPUs) are intentionally dropped
    because SLURM controls those on cluster nodes.
    """

    def __init__(self, sif_dir: str = ".") -> None:
        """Args:
            sif_dir: Directory that contains the .sif files.
        """
        self.sif_dir = Path(sif_dir)

    def _image_to_sif(self, image: str) -> Path:
        """Derive a .sif filename from a Docker image reference.

        "cellsimbench/sclambda:latest" -> <sif_dir>/sclambda.sif
        """
        name = image.split("/")[-1].split(":")[0]
        return self.sif_dir / f"{name}.sif"

    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, Dict[str, str]],
        docker_config: Dict[str, Any],
        container_name: str = "cellsimbench",
        environment: Optional[Dict[str, str]] = None,
        gpu_id: Optional[int] = None,
    ) -> None:
        """Run an Apptainer container with the given configuration.

        Args:
            image: Docker image name used to locate the .sif file.
            command: Arguments forwarded to the container's runscript.
            volumes: Docker-style volume dict {host_path: {bind, mode}}.
            docker_config: Docker settings; only `gpu` key is used.
            container_name: Label used in log messages.
            environment: Environment variables to pass into the container.
            gpu_id: If set, restricts CUDA_VISIBLE_DEVICES to this GPU.

        Raises:
            FileNotFoundError: If the .sif file does not exist.
            RuntimeError: If the container exits with a non-zero status.
        """
        sif_path = self._image_to_sif(image)
        if not sif_path.exists():
            raise FileNotFoundError(
                f"Apptainer SIF image not found: {sif_path}\n"
                f"Pull it with:\n"
                f"  apptainer pull {sif_path} docker://millerh1/cellsimbench-{sif_path.stem}:latest"
            )

        cmd: List[str] = ["apptainer", "run"]

        # --writable-tmpfs gives the container a writable overlay so models
        # that write to paths inside the image (e.g. /app/data in GEARS) work.
        # --no-mount tmp prevents Apptainer from bind-mounting the host /tmp
        # over the container's /tmp, which would hide editable pip installs
        # placed there during the Docker build.
        cmd += ["--writable-tmpfs", "--no-mount", "tmp"]

        # GPU passthrough
        env = dict(environment) if environment else {}
        if docker_config.get("gpu", True):
            cmd.append("--nv")
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Volume binds
        for host_path, mount in volumes.items():
            bind_str = f"{host_path}:{mount['bind']}"
            if mount.get("mode") == "ro":
                bind_str += ":ro"
            cmd += ["--bind", bind_str]

        # Environment variables
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]

        cmd.append(str(sif_path))
        cmd.extend(command)

        log.info(f"Starting {container_name} with Apptainer image: {sif_path.name}")
        log.info(f"Volume mounts:")
        for host_path, mount in volumes.items():
            log.info(f"  {host_path} -> {mount['bind']}")
        log.info(f"Full command: {' '.join(cmd)}")

        # Stream output directly to the caller's stdout/stderr so it appears
        # in SLURM job logs without extra buffering.
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            raise RuntimeError(
                f"{container_name} failed with exit code {result.returncode}\n"
                f"Command: {' '.join(command)}\n"
                f"Image: {sif_path}\n\n"
                f"Debugging hints:\n"
                f"- Check that all bind paths exist on the host\n"
                f"- Verify the SIF image is intact (try: apptainer inspect {sif_path})\n"
                f"- Check configuration file format and values\n"
                f"- Ensure sufficient memory/disk space is available"
            )

        log.info(f"{container_name} completed successfully")
