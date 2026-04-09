"""vLLM server management: start, stop, status.

Runs vLLM in a container (podman/docker). Just start/stop/status.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import signal
import socket
import subprocess
from pathlib import Path

from loguru import logger
from sciwrite_lint.config import LintConfig

# Persistent state
_STATE_DIR = Path.home() / ".sciwrite-lint"
_LAST_PORTS_FILE = _STATE_DIR / "last_ports.json"

# Model profiles: HuggingFace model ID, served name, and config
MODELS: dict[str, dict] = {
    "qwen3": {
        "hf_model": "RedHatAI/Qwen3-8B-FP8-dynamic",
        "served_name": "qwen3-8b-fp8",
        "reasoning_parser": "qwen3",  # server-side <think> parsing
    },
    "gemma3": {
        "hf_model": "RedHatAI/gemma-3-12b-it-FP8-dynamic",
        "served_name": "gemma3-12b-fp8",
        "reasoning_parser": "",  # no thinking support
    },
}

# Default vLLM flags for consumer-GPU deployment
_DEFAULT_GPU_MEM = 0.9
_DEFAULT_MAX_MODEL_LEN = 40_960
_DEFAULT_PORT = 5001


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------


def _load_last_ports() -> dict[str, int]:
    """Load remembered ports from ~/.sciwrite-lint/last_ports.json."""
    try:
        return json.loads(_LAST_PORTS_FILE.read_text())  # type: ignore[no-any-return]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_last_port(service: str, port: int) -> None:
    """Remember which port a service last used."""
    ports = _load_last_ports()
    if ports.get(service) == port:
        return
    ports[service] = port
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_PORTS_FILE.write_text(json.dumps(ports, indent=2) + "\n")
    except OSError:
        pass


def _port_available(host: str, port: int) -> bool:
    """Check whether port can be bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _resolve_port(config: LintConfig) -> int:
    """Resolve vLLM port: config endpoint > remembered > default."""
    # Parse port from config endpoint
    from urllib.parse import urlparse

    parsed = urlparse(config.llm_endpoint)
    config_port = parsed.port or _DEFAULT_PORT

    # Try remembered port
    last = _load_last_ports().get("vllm")
    if last is not None and last != config_port and _port_available("0.0.0.0", last):
        return last
    return config_port


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


async def _check_api_health(endpoint: str) -> dict | None:
    """Check if vLLM API is responding. Returns model list or None."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{endpoint}/models", timeout=5.0)
            if resp.status_code == 200:
                return resp.json()  # type: ignore[no-any-return]
    except httpx.HTTPError:
        pass
    return None


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _detect_container_runtime() -> str | None:
    """Detect available container runtime (podman preferred)."""
    for rt in ("podman", "docker"):
        if shutil.which(rt):
            return rt
    return None


def _container_name(model: str) -> str:
    return f"sciwrite-lint-vllm-{model}"


def _container_exists(runtime: str, name: str) -> bool:
    result = subprocess.run(
        [runtime, "container", "inspect", name],
        capture_output=True,
    )
    return result.returncode == 0


def _container_running(runtime: str, name: str) -> bool:
    result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


# ---------------------------------------------------------------------------
# Containerized vLLM
# ---------------------------------------------------------------------------


def start_container(
    config: LintConfig,
    model: str | None = None,
    pull: bool = False,
) -> int:
    """Start vLLM in a container (podman/docker)."""
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    model_name = model or config.llm_model
    profile = MODELS.get(model_name)
    if not profile:
        logger.error(f"Unknown model '{model_name}'. Available: {', '.join(MODELS)}")
        return 1

    name = _container_name(model_name)
    serve_port = _resolve_port(config)

    # Already running?
    if _container_running(runtime, name):
        logger.info(f"Container '{name}' is already running")
        logger.info("Use 'sciwrite-lint vllm restart' to restart")
        return 0

    # Remove stopped container (config may have changed)
    if _container_exists(runtime, name):
        logger.info(f"Removing stopped container '{name}'")
        subprocess.run([runtime, "rm", "-f", name], capture_output=True)

    # Pull latest image
    image = config.vllm_image
    if pull:
        logger.info(f"Pulling {image}")
        pull_result = subprocess.run([runtime, "pull", image])
        if pull_result.returncode != 0:
            logger.error(f"Failed to pull image: {image}")
            return 1

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        name,
        "--device",
        "nvidia.com/gpu=all",
        "--memory",
        config.vllm_memory,
        "-p",
        f"{serve_port}:8000",
        "-v",
        f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface",
        "--restart",
        "unless-stopped",
        image,
        profile["hf_model"],  # positional arg (--model deprecated in vLLM v0.13)
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        profile["served_name"],
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(_DEFAULT_GPU_MEM),
        "--max-model-len",
        str(_DEFAULT_MAX_MODEL_LEN),
        "--kv-cache-dtype",
        "fp8",
        "--enable-chunked-prefill",
    ]

    # Server-side reasoning parser for thinking models (Qwen3)
    if profile.get("reasoning_parser"):
        cmd.extend(["--reasoning-parser", profile["reasoning_parser"]])

    logger.info(f"Starting vLLM container '{name}'")
    logger.info(f"Runtime: {runtime}")
    logger.info(f"Image: {image}")
    logger.info(f"Model: {profile['hf_model']}")
    logger.info(f"Served as: {profile['served_name']}")
    logger.info(f"Port: {serve_port}")
    logger.info(f"GPU memory: {_DEFAULT_GPU_MEM}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "CDI" in stderr or "unresolvable" in stderr:
            logger.error(
                "GPU passthrough not configured (CDI spec missing).\n\n"
                "Fix with:\n"
                "  sudo apt install nvidia-container-toolkit\n"
                "  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml\n"
                "  podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi"
            )
        else:
            logger.error(f"Failed to start container:\n{stderr}")
        return 1

    container_id = result.stdout.strip()[:12]
    logger.info(f"Container started: {container_id}")
    logger.info(f"API will be available at http://localhost:{serve_port}/v1")

    # Check if model is cached
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = f"models--{profile['hf_model'].replace('/', '--')}"
    if (hf_cache / model_cache_name).exists():
        logger.info("Model weights cached. Loading into GPU (~30-60s)")
    else:
        logger.info(
            "First run: downloading model weights. This may take several minutes"
        )
    logger.info(f"Check logs: {runtime} logs -f {name}")
    logger.info("Check status: sciwrite-lint vllm status")

    _save_last_port("vllm", serve_port)
    return 0


def stop_container(config: LintConfig, model: str | None = None) -> int:
    """Stop the vLLM container."""
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    model_name = model or config.llm_model
    name = _container_name(model_name)

    if not _container_exists(runtime, name):
        logger.info(f"Container '{name}' does not exist")
        return 0

    if not _container_running(runtime, name):
        logger.info(f"Container '{name}' is not running")
        return 0

    logger.info(f"Stopping container '{name}'")
    result = subprocess.run([runtime, "stop", name], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to stop container:\n{result.stderr.strip()}")
        return 1
    logger.info("Container stopped. GPU memory released")
    return 0


def container_status(config: LintConfig, model: str | None = None) -> int:
    """Show vLLM container status and API health."""
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    model_name = model or config.llm_model
    name = _container_name(model_name)

    if not _container_exists(runtime, name):
        logger.info(f"Container '{name}' does not exist")
        logger.info("Start it with: sciwrite-lint containers start")
        return 0

    inspect_result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
    )
    state = (
        inspect_result.stdout.strip() if inspect_result.returncode == 0 else "unknown"
    )
    logger.info(f"Container: {name}")
    logger.info(f"State: {state}")

    if state != "running":
        logger.info("API: not available (container not running)")
        return 0

    # API health
    health = asyncio.run(_check_api_health(config.llm_endpoint))
    if health:
        models = [m["id"] for m in health.get("data", [])]
        logger.info(f"API: ready (models: {', '.join(models)})")
    else:
        logger.info(f"API: loading (not responding yet at {config.llm_endpoint})")

    return 0


def container_logs(
    config: LintConfig,
    model: str | None = None,
    follow: bool = False,
    tail: int = 50,
) -> int:
    """Show vLLM container logs."""
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    model_name = model or config.llm_model
    name = _container_name(model_name)

    if not _container_exists(runtime, name):
        logger.error(f"Container '{name}' does not exist")
        return 1

    cmd = [runtime, "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(name)

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
    return 0


def remove_container(
    config: LintConfig,
    model: str | None = None,
    force: bool = False,
) -> int:
    """Remove the vLLM container."""
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    model_name = model or config.llm_model
    name = _container_name(model_name)

    if not _container_exists(runtime, name):
        logger.info(f"Container '{name}' does not exist")
        return 0

    if _container_running(runtime, name) and not force:
        logger.info(f"Container '{name}' is still running")
        logger.info("Stop it first with 'sciwrite-lint vllm stop', or use --force")
        return 1

    cmd = [runtime, "rm"]
    if force:
        cmd.append("-f")
    cmd.append(name)

    logger.info(f"Removing container '{name}'")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to remove container:\n{result.stderr.strip()}")
        return 1
    logger.info("Container removed")
    return 0


# ---------------------------------------------------------------------------
# Status (unified — checks both native and container)
# ---------------------------------------------------------------------------


def status(config: LintConfig) -> int:
    """Show vLLM status: check API health regardless of how it was started."""
    endpoint = config.llm_endpoint
    logger.info(f"vLLM endpoint: {endpoint}")

    # Check API
    health = asyncio.run(_check_api_health(endpoint))
    if health:
        models = [m["id"] for m in health.get("data", [])]
        logger.info("Status: ready")
        logger.info(f"Models: {', '.join(models)}")
    else:
        logger.info("Status: not responding")
        logger.info("Start with: sciwrite-lint containers start")

    # Check for running containers
    runtime = _detect_container_runtime()
    if runtime:
        for model_name in MODELS:
            name = _container_name(model_name)
            if _container_running(runtime, name):
                logger.info(f"Container '{name}' is running ({runtime})")

    # Check for native process
    result = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True)
    if result.returncode == 0:
        logger.info("Native vLLM process detected")

    return 0
