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
        "kind": "text",
        "gpu_memory_utilization": 0.85,
    },
    "gemma3": {
        "hf_model": "RedHatAI/gemma-3-12b-it-FP8-dynamic",
        "served_name": "gemma3-12b-fp8",
        "reasoning_parser": "",  # no thinking support
        "kind": "text",
        # gemma-3 attention (logit soft-capping) is incompatible with an
        # fp8 KV cache on this vLLM build — EngineCore crash-loops during
        # init (weights load, then crash at CUDA-graph capture) and the
        # API never comes up. Pin the KV cache to auto and disable CUDA
        # graph capture; this is the confirmed-working combination on Ada.
        # Qwen3 keeps the fp8 default and CUDA graphs unchanged.
        "kv_cache_dtype": "auto",
        "enforce_eager": True,
    },
    "qwen3-vl": {
        "hf_model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "served_name": "qwen3-vl-8b-fp8",
        "reasoning_parser": "",
        "kind": "vision",
        "port": 5002,
        "max_model_len": 8192,
        "memory": "8g",  # vision models need more host RAM for image preprocessing
        # Match text vLLM (0.85) instead of the 0.9 default. At 0.9 on
        # a 20 GB card the engine reserves 18 GB, leaving only ~2 GB
        # headroom for prefill activations + PyTorch caching allocator
        # drift. Under high-concurrency vision (50+ image requests
        # arriving together) the activations spill into shared GPU
        # memory via PCIe, which is 30-100× slower than dedicated VRAM.
        # 0.85 = 17 GB reservation + 3 GB headroom keeps activations
        # on-device during the heaviest concurrent prefill bursts.
        "gpu_memory_utilization": 0.85,
        # ``--max-num-seqs`` for the vision container is derived from
        # ``config.vision_server_max_seqs`` at start-time (see
        # ``start_container``). The client-side cap
        # (``vision_max_concurrency`` — controller upper_bound or static
        # semaphore) is a separate field; they're usually set to the
        # same value but have distinct roles.
        #
        # Empirically (Qwen3-VL-8B-FP8, 20GB card, max_model_len=8192),
        # varying ``--max-num-seqs`` from 16 → 64 → 256 leaves
        # ``num_gpu_blocks`` unchanged at 4986. So the activation-slot
        # budget does NOT measurably steal from the KV pool in this
        # configuration. Set this as high as the workload demands; the
        # dynamic concurrency controller (see
        # ``sciwrite_lint/llm/concurrency_optimizer/``) decides the
        # actual in-flight count from observed KV utilization.
    },
}

# Convenience sets for CLI choices
TEXT_MODELS = [k for k, v in MODELS.items() if v["kind"] == "text"]
VISION_MODELS = [k for k, v in MODELS.items() if v["kind"] == "vision"]

# Default vLLM flags for consumer-GPU deployment
_DEFAULT_GPU_MEM = 0.9
_DEFAULT_MAX_MODEL_LEN = 40_960
_DEFAULT_PORT = 5001
_DEFAULT_VISION_PORT = 5002
# fp8 KV cache halves KV-pool memory pressure and is the right default on
# Ada (native FP8). Profiles whose attention is incompatible with an fp8 KV
# cache override this per-profile via ``kv_cache_dtype`` (see ``gemma3``).
_DEFAULT_KV_CACHE_DTYPE = "fp8"


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


def _resolve_port(config: LintConfig, model_name: str | None = None) -> int:
    """Resolve vLLM port: profile override > config endpoint > remembered > default."""
    # Vision models have a fixed port in their profile
    if model_name:
        profile = MODELS.get(model_name)
        if profile and "port" in profile:
            return profile["port"]

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


async def wait_for_ready(endpoint: str, timeout: int = 300, interval: int = 5) -> bool:
    """Poll vLLM API until it responds or timeout is reached."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = await _check_api_health(endpoint)
        if health:
            return True
        await asyncio.sleep(interval)
    return False


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


def _identify_port_holder(runtime: str | None, port: int) -> str:
    """Try to identify which container holds a port. Returns ' by <name>' or ''."""
    if not runtime:
        return ""
    result = subprocess.run(
        [runtime, "ps", "--format", "{{.Names}}\t{{.Ports}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    for line in result.stdout.strip().splitlines():
        if f":{port}->" in line:
            name = line.split("\t")[0]
            return f" by container '{name}'"
    return ""


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


def _container_cmd_args(runtime: str, name: str) -> list[str]:
    """Return the ``Config.Cmd`` of an existing container, or [] on error.

    Used by the start-time drift detector — args are baked at
    container creation, so a config bump doesn't propagate to a
    container that's merely restarted.
    """
    import json as _json

    result = subprocess.run(
        [runtime, "inspect", name, "--format", "{{json .Config.Cmd}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    try:
        parsed = _json.loads(result.stdout.strip())
        return list(parsed) if parsed else []
    except (_json.JSONDecodeError, TypeError):
        return []


def _expected_max_num_seqs(config: "LintConfig", profile: dict) -> int:
    """Mirror of the value passed to ``--max-num-seqs`` at create time."""
    if profile.get("kind") == "vision":
        return int(config.vision_server_max_seqs)
    return int(config.llm_server_max_seqs)


def _parse_max_num_seqs(args: list[str]) -> int | None:
    """Extract the integer value of ``--max-num-seqs`` from Cmd args.

    Returns ``None`` when the flag is absent or its value isn't a
    base-10 int. The single source of truth for both the start-time
    drift detector and the runtime-clamp helper.
    """
    for i, a in enumerate(args):
        if a == "--max-num-seqs" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


def get_running_max_num_seqs(
    config: LintConfig, model: str | None = None
) -> int | None:
    """Read ``--max-num-seqs`` from the running vLLM container.

    Returns the integer cap baked at ``podman run`` time, or ``None``
    when nothing can be probed (no runtime, container isn't running,
    inspect fails, arg missing or unparseable). Best-effort — callers
    treat ``None`` as "couldn't determine" and skip the clamp.
    """
    runtime = _detect_container_runtime()
    if not runtime:
        return None
    name = _container_name(model or config.llm_model)
    if not _container_running(runtime, name):
        return None
    return _parse_max_num_seqs(_container_cmd_args(runtime, name))


def effective_max_concurrency(
    config: LintConfig,
    requested_cap: int,
    *,
    model: str | None = None,
    label: str = "text",
) -> int:
    """Clamp ``requested_cap`` to the running container's ``--max-num-seqs``.

    The controller's effective upper bound is the smaller of the
    application-side cap (``requested_cap`` from
    ``llm_max_concurrency`` / ``vision_max_concurrency``) and the
    server-side admission ceiling (``--max-num-seqs`` baked into the
    running container). Pushing past the server cap is silent —
    requests pile up in vLLM's waiting queue, latency degrades, and
    the engine has been observed to crash under sustained pressure
    (see ``_check_container_arg_drift`` for the related drift signal).

    On drift the function logs at ERROR (the operator must recreate
    the container to lift the cap) and returns ``server_cap`` so the
    controller's ``upper_bound`` tracks reality. When the server cap
    can't be determined returns ``requested_cap`` unchanged — drift
    detection then moves to the next sanctioned ``start_container``
    call.
    """
    server_cap = get_running_max_num_seqs(config, model=model)
    if server_cap is None or server_cap >= requested_cap:
        return requested_cap
    logger.error(
        "vLLM container --max-num-seqs={} below client cap={} for {}. "
        "Capping client at server limit so the controller never pushes "
        "past it; recreate the container with `sciwrite-lint containers "
        "start` to apply current config.",
        server_cap,
        requested_cap,
        label,
    )
    return server_cap


def _check_container_arg_drift(
    runtime: str,
    name: str,
    config: "LintConfig",
    profile: dict,
) -> bool:
    """Detect drift between the container's baked args and current config.

    Args are baked at ``podman run`` time. ``podman start`` of an
    existing stopped container reuses the original args silently. So
    bumping a config value (e.g. ``llm_max_concurrency`` 12 → 256) and
    just restarting yields a container that *looks* healthy but
    silently throttles at the old value, which has caused observable
    overload-induced engine crashes mid-run.

    Returns True when drift is detected (caller is expected to recreate
    the container). Returns False when args match or detection itself
    fails (e.g. ``podman inspect`` errors out for an unrelated reason —
    not the right place to abort the whole call). Logs an ERROR on
    drift so the recreate is loud rather than silent.
    """
    args = _container_cmd_args(runtime, name)
    if not args:
        return False

    drifts: list[str] = []
    actual = _parse_max_num_seqs(args)
    expected = _expected_max_num_seqs(config, profile)
    if actual is not None and actual != expected:
        drifts.append(f"--max-num-seqs: container={actual}, config={expected}")

    if drifts:
        logger.error(
            "Container '{}' has stale args from a previous create — "
            "config bumps did not apply:\n  {}\n"
            "Auto-recreating to apply current config (running with stale "
            "args caused silent overload crashes; recreating is mandatory).",
            name,
            "\n  ".join(drifts),
        )
        return True
    return False


def start_container(
    config: LintConfig,
    model: str | None = None,
    pull: bool = False,
    replace: bool = False,
) -> int:
    """Start vLLM in a container (podman/docker).

    With another vLLM model's container already running, both end up
    fighting for VRAM: each launches with ``--gpu-memory-utilization
    0.85`` but vLLM scales the second one down to whatever's free at
    startup, leaving both with crippled KV cache budgets. To prevent
    that silent degradation, this function refuses to start when a
    different vLLM model is running unless ``replace=True``, in which
    case the conflicting container(s) are stopped first.
    """
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
    serve_port = _resolve_port(config, model_name)

    # Probe state once and reuse — each helper shells out to the
    # container runtime, so collapsing the calls saves both wall time
    # and avoids the logic-split that comes with re-asking the runtime
    # in every branch.
    target_running = _container_running(runtime, name)
    target_exists = target_running or _container_exists(runtime, name)
    drift = target_exists and _check_container_arg_drift(runtime, name, config, profile)

    # Conflict check: refuse to coexist with another vLLM model unless
    # ``replace=True``. Skipped on the idempotent path (target already
    # running with matching args) — a no-op rerun shouldn't error out.
    if not (target_running and not drift):
        conflicting = [
            (other_model, _container_name(other_model))
            for other_model in MODELS
            if other_model != model_name
            and _container_running(runtime, _container_name(other_model))
        ]
        if conflicting:
            if replace:
                for other_model, other_name in conflicting:
                    logger.info(
                        "Stopping vLLM '{}' to free GPU for '{}'",
                        other_name,
                        name,
                    )
                    subprocess.run([runtime, "stop", other_name], capture_output=True)
            else:
                others_models = ", ".join(om for om, _ in conflicting)
                first_other = conflicting[0][0]
                logger.error(
                    f"Cannot start vLLM '{model_name}': another vLLM is "
                    f"already running ({others_models}). Two vLLM containers "
                    f"can't share the GPU safely — both end up with crippled "
                    f"KV cache.\n"
                    f"  - Swap automatically: "
                    f"sciwrite-lint vllm start --model {model_name} --replace\n"
                    f"  - Or stop the other manually: "
                    f"sciwrite-lint vllm stop --model {first_other}"
                )
                return 1

    if target_running:
        if drift:
            logger.info(
                f"Stopping running container '{name}' to recreate with current args"
            )
            subprocess.run([runtime, "stop", name], capture_output=True)
            subprocess.run([runtime, "rm", "-f", name], capture_output=True)
            # fall through to recreate with current config args
        else:
            logger.info(f"Container '{name}' is already running")
            logger.info("Use 'sciwrite-lint vllm restart' to restart")
            return 0

    # Restart stopped container (fast — weights cached in page cache)
    elif target_exists and not pull:
        if drift:
            logger.info(
                f"Removing stopped container '{name}' to recreate with current args"
            )
            subprocess.run([runtime, "rm", "-f", name], capture_output=True)
            # fall through to recreate with current config args
        else:
            logger.info(f"Restarting stopped container '{name}'")
            result = subprocess.run(
                [runtime, "start", name], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"Container restarted: {name}")
                logger.info(
                    f"API will be available at http://localhost:{serve_port}/v1"
                )
                port_key = "vllm-vision" if profile.get("kind") == "vision" else "vllm"
                _save_last_port(port_key, serve_port)
                return 0
            # Start failed (e.g. config changed) — fall through to rm + run
            logger.info(f"Restart failed, recreating container '{name}'")
            subprocess.run([runtime, "rm", "-f", name], capture_output=True)

    # Check port is free before attempting to start
    if not _port_available("0.0.0.0", serve_port):
        blocker = _identify_port_holder(runtime, serve_port)
        logger.error(
            f"Port {serve_port} is already in use{blocker}.\n"
            f"Free it with: sciwrite-lint containers stop\n"
            f"Or check manually: ss -tlnp | grep {serve_port}"
        )
        return 1

    # Pull latest image
    image = config.vllm_image
    if pull:
        logger.info(f"Pulling {image}")
        pull_result = subprocess.run([runtime, "pull", image])
        if pull_result.returncode != 0:
            logger.error(f"Failed to pull image: {image}")
            return 1

    max_model_len = profile.get("max_model_len", _DEFAULT_MAX_MODEL_LEN)
    gpu_mem = profile.get("gpu_memory_utilization", _DEFAULT_GPU_MEM)
    container_memory = profile.get("memory", config.vllm_memory)
    kv_cache_dtype = profile.get("kv_cache_dtype", _DEFAULT_KV_CACHE_DTYPE)
    enforce_eager = profile.get("enforce_eager", False)

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        name,
        "--device",
        "nvidia.com/gpu=all",
        "--memory",
        container_memory,
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
        str(gpu_mem),
        "--max-model-len",
        str(max_model_len),
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--enable-chunked-prefill",
        # Per-step compute budget for prefill chunks + decode tokens
        # combined. 8192 because:
        #   1. ``full_paper_consistency`` sends ~26k-token system
        #      prompts. With prefix caching (default-on in v0.18+),
        #      only the first check pays the cold prefill; subsequent
        #      checks hit cached KV. 8192 keeps that cold prefill at
        #      ~3 iterations rather than ~13 at budget=2048.
        #   2. KV pool is unaffected at our VRAM reservation;
        #      activation headroom stays comfortable.
        #   3. Headroom against longer prompts running concurrently
        #      with short ones, where prefill could starve decode
        #      under a tighter budget.
        # Sizing for the full 26k case (e.g. 32768) burns ~4× iter
        # latency for no measured win and tightens activation memory
        # headroom inside ``--gpu-memory-utilization=0.85``.
        "--max-num-batched-tokens",
        "8192",
    ]

    # ``--max-num-seqs`` is the server-side admission ceiling. It is
    # the **server-side** knob ``*_server_max_seqs``, distinct from the
    # **client-side** ``*_max_concurrency`` that drives the controller's
    # ``upper_bound`` / static semaphore cap. The two are usually set to
    # the same value but their roles are different, hence separate
    # fields — see ``LintConfig`` docstrings.
    if profile["kind"] == "vision":
        cmd.extend(["--max-num-seqs", str(config.vision_server_max_seqs)])
    else:
        cmd.extend(["--max-num-seqs", str(config.llm_server_max_seqs)])

    # Disable CUDA graph capture for profiles that crash during capture
    # (e.g. gemma-3 on this vLLM build). Off by default — CUDA graphs are
    # a meaningful decode-latency win when the model supports them.
    if enforce_eager:
        cmd.append("--enforce-eager")

    # Server-side reasoning parser for thinking models (Qwen3)
    if profile.get("reasoning_parser"):
        cmd.extend(["--reasoning-parser", profile["reasoning_parser"]])

    logger.info(f"Starting vLLM container '{name}'")
    logger.info(f"Runtime: {runtime}")
    logger.info(f"Image: {image}")
    logger.info(f"Model: {profile['hf_model']}")
    logger.info(f"Served as: {profile['served_name']}")
    logger.info(f"Port: {serve_port}")
    logger.info(f"GPU memory: {gpu_mem}")

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

    port_key = "vllm-vision" if profile.get("kind") == "vision" else "vllm"
    _save_last_port(port_key, serve_port)
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
