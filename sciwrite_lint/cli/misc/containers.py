"""``sciwrite-lint containers`` — manage GROBID + vLLM containers together."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path

from sciwrite_lint.config import load_config


def run_containers(args: argparse.Namespace) -> int:
    """Manage both GROBID and vLLM containers together."""
    from sciwrite_lint.cli.misc._monitor import (
        _fetch_vllm_metrics,
        _print_container_logs,
        _run_containers_monitor,
    )
    from sciwrite_lint.pdf.grobid import (
        CONTAINER_NAME as GROBID_CONTAINER,
        CONTAINER_RUNTIME,
        container_memory_status,
        gpu_memory_status,
        is_grobid_running,
        start_grobid,
        stop_grobid,
    )
    from sciwrite_lint.vllm.vllm_server import (
        VISION_MODELS,
        _check_api_health,
        _container_name as vllm_container_name,
        _container_running,
        _detect_container_runtime,
        start_container,
        stop_container,
    )

    config = load_config(
        Path(args.config) if hasattr(args, "config") and args.config else None
    )
    action = args.action

    if action == "status":
        runtime = _detect_container_runtime()
        grobid_up = asyncio.run(is_grobid_running())
        if grobid_up:
            mem = container_memory_status(CONTAINER_RUNTIME, GROBID_CONTAINER)
            suffix = f"  RAM: {mem}" if mem else ""
            print(f"GROBID:  running at http://localhost:8070{suffix}")
        else:
            print("GROBID:  not running")

        endpoint = config.llm_endpoint
        health = asyncio.run(_check_api_health(endpoint))
        if health:
            models = [m["id"] for m in health.get("data", [])]
            vllm_name = vllm_container_name(config.llm_model)
            mem = container_memory_status(runtime, vllm_name) if runtime else None
            mem_str = f"  RAM: {mem}" if mem else ""
            print(
                f"vLLM (text):   running at {endpoint} ({', '.join(models)}){mem_str}"
            )
            vram = gpu_memory_status()
            metrics = _fetch_vllm_metrics(endpoint)
            if vram:
                used_gb = vram[0] / (1024**3)
                total_gb = vram[1] / (1024**3)
                vram_str = (
                    f"{used_gb:.1f}GB / {total_gb:.1f}GB ({vram[0] / vram[1]:.0%})"
                )
            else:
                vram_str = None
            gpu_parts = [f"VRAM: {vram_str}"] if vram_str else []
            if metrics:
                gpu_parts.append(metrics)
            if gpu_parts:
                print(f"               {', '.join(gpu_parts)}")
        else:
            print("vLLM (text):   not running")

        runtime = _detect_container_runtime()
        vision_up = False
        for vm in VISION_MODELS:
            vm_name = vllm_container_name(vm)
            if runtime and _container_running(runtime, vm_name):
                from sciwrite_lint.vllm.vllm_server import MODELS

                vm_profile = MODELS[vm]
                vm_port = vm_profile.get("port", 5002)
                vm_endpoint = f"http://localhost:{vm_port}/v1"
                vm_health = asyncio.run(_check_api_health(vm_endpoint))
                if vm_health:
                    vm_models = [m["id"] for m in vm_health.get("data", [])]
                    mem = container_memory_status(runtime, vm_name) if runtime else None
                    mem_str = f"  RAM: {mem}" if mem else ""
                    print(
                        f"vLLM (vision): running at {vm_endpoint}"
                        f" ({', '.join(vm_models)}){mem_str}"
                    )
                else:
                    print("vLLM (vision): loading (container up, API not ready)")
                vision_up = True
            else:
                print("vLLM (vision): not running")

        print()
        print("Commands:")
        if not grobid_up or not health:
            print(
                "  sciwrite-lint containers start            # start GROBID + text vLLM"
            )
        if not vision_up:
            print(
                "  sciwrite-lint containers start --vision   # also start vision vLLM"
            )
        print("  sciwrite-lint containers stop             # stop all")
        print("  sciwrite-lint grobid start|stop|status    # manage GROBID alone")
        print("  sciwrite-lint vllm start|stop|status      # manage vLLM alone")
        print("  sciwrite-lint vllm logs [-f]              # follow vLLM logs")

        if runtime:
            log_containers = [
                ("GROBID", GROBID_CONTAINER),
                ("vLLM (text)", vllm_container_name(config.llm_model)),
            ]
            for vm in VISION_MODELS:
                log_containers.append(("vLLM (vision)", vllm_container_name(vm)))
            for label, name in log_containers:
                result = subprocess.run(
                    [runtime, "container", "inspect", name],
                    capture_output=True,
                )
                if result.returncode != 0:
                    continue
                print(f"\n{'─' * 60}")
                print(f"{label} logs (last 15 lines):")
                print(f"{'─' * 60}")
                _print_container_logs(runtime, name, tail=15)

        return 0

    elif action == "start":
        failed = False
        update = getattr(args, "update", False)
        vision = getattr(args, "vision", False)

        if update:
            print(f"Pulling GROBID image: {config.grobid_image}")
            subprocess.run([CONTAINER_RUNTIME, "pull", config.grobid_image])

        print(f"Starting GROBID container (memory limit: {config.grobid_memory})...")
        if asyncio.run(
            start_grobid(memory=config.grobid_memory, image=config.grobid_image)
        ):
            print("GROBID: running at http://localhost:8070")
        else:
            print("GROBID: failed to start within 60s")
            failed = True

        model = getattr(args, "model", None)
        ret = start_container(config, model=model, pull=update)
        if ret != 0:
            failed = True

        if vision:
            for vm in VISION_MODELS:
                ret = start_container(config, model=vm, pull=update)
                if ret != 0:
                    failed = True

        return 1 if failed else 0

    elif action == "stop":
        stop_grobid()
        print("GROBID: stopped")
        stop_container(config, model=getattr(args, "model", None))
        for vm in VISION_MODELS:
            stop_container(config, model=vm)
        return 0

    elif action == "restart":
        model = getattr(args, "model", None)
        recreate = getattr(args, "recreate", False)
        runtime = _detect_container_runtime()

        stop_grobid()
        stop_container(config, model=model)
        for vm in VISION_MODELS:
            stop_container(config, model=vm)

        if recreate and runtime:
            print("Removing containers to apply current config...")
            subprocess.run(
                [runtime, "rm", GROBID_CONTAINER],
                capture_output=True,
            )
            vllm_name = vllm_container_name(config.llm_model)
            subprocess.run(
                [runtime, "rm", vllm_name],
                capture_output=True,
            )
            for vm in VISION_MODELS:
                subprocess.run(
                    [runtime, "rm", vllm_container_name(vm)],
                    capture_output=True,
                )

        print("Containers stopped. Restarting...")
        args.action = "start"
        return run_containers(args)

    elif action == "monitor":
        return _run_containers_monitor(config, interval=getattr(args, "interval", 2))

    return 0
