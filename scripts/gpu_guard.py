#!/usr/bin/env python3
"""Reserve an idle GPU briefly or while running a command.

This is intended for shared lab servers where a job can spend a long time in
Python imports before it creates a CUDA context. The guard checks that a GPU is
idle, creates a small CUDA allocation so the GPU is visibly occupied, and then
runs the requested command with CUDA_VISIBLE_DEVICES set to that GPU.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    bus_id: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization: int


def run_nvidia_smi(args: list[str]) -> str:
    return subprocess.check_output(["nvidia-smi", *args], text=True).strip()


def get_gpu_info(gpu: int) -> GpuInfo:
    output = run_nvidia_smi(
        [
            "--query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        if int(parts[0]) == gpu:
            return GpuInfo(
                index=int(parts[0]),
                bus_id=parts[1],
                name=parts[2],
                memory_used_mb=int(parts[3]),
                memory_total_mb=int(parts[4]),
                utilization=int(parts[5]),
            )
    raise SystemExit(f"GPU {gpu} was not found by nvidia-smi")


def get_compute_apps(bus_id: str) -> list[str]:
    try:
        output = run_nvidia_smi(
            [
                "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return []
    apps = []
    for line in output.splitlines():
        if line.strip().startswith(bus_id):
            apps.append(line.strip())
    return apps


def check_idle(gpu: int, max_used_mb: int, max_utilization: int) -> GpuInfo:
    info = get_gpu_info(gpu)
    apps = get_compute_apps(info.bus_id)
    if apps or info.memory_used_mb > max_used_mb or info.utilization > max_utilization:
        print(
            "GPU is not idle:\n"
            f"  gpu={info.index} bus={info.bus_id} name={info.name}\n"
            f"  memory={info.memory_used_mb}/{info.memory_total_mb} MiB "
            f"utilization={info.utilization}%\n"
            f"  compute_apps={apps if apps else 'none'}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(3)
    return info


def wait_until_idle(
    gpu: int,
    max_used_mb: int,
    max_utilization: int,
    wait_timeout: int,
    poll_interval: int,
) -> GpuInfo:
    deadline = time.time() + wait_timeout
    while True:
        try:
            return check_idle(gpu, max_used_mb, max_utilization)
        except SystemExit as exc:
            if wait_timeout <= 0 or time.time() >= deadline:
                raise
            time.sleep(max(1, poll_interval))


def allocate_guard_tensor(gpu: int, hold_mb: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available after setting CUDA_VISIBLE_DEVICES")
    device_name = torch.cuda.get_device_name(0)
    num_bytes = max(1, hold_mb) * 1024 * 1024
    tensor = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    tensor.fill_(1)
    torch.cuda.synchronize()
    print(
        f"Reserved GPU {gpu} as visible cuda:0 ({device_name}) "
        f"with {hold_mb} MiB guard allocation.",
        file=sys.stderr,
        flush=True,
    )
    return tensor


def wait_forever() -> int:
    stop = False

    def handle_signal(signum, _frame):
        nonlocal stop
        print(f"Received signal {signum}; releasing GPU guard.", file=sys.stderr, flush=True)
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    while not stop:
        time.sleep(5)
    return 0


def run_command(command: list[str]) -> int:
    env = os.environ.copy()
    child = subprocess.Popen(command, env=env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT)
        return child.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a GPU is idle, reserve it with a small CUDA allocation, then run a command."
    )
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index from nvidia-smi.")
    parser.add_argument(
        "--hold-mb",
        type=int,
        default=512,
        help="Small guard allocation size. This is a signal/reservation, not an attempt to fill the GPU.",
    )
    parser.add_argument(
        "--max-used-mb",
        type=int,
        default=256,
        help="Refuse to reserve if nvidia-smi reports more memory already used on the GPU.",
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=5,
        help="Refuse to reserve if GPU utilization is above this percent.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=0,
        help="Seconds to keep polling for an idle GPU before refusing. Default refuses immediately.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between idle checks when --wait-timeout is positive.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'. If omitted, hold the GPU until interrupted.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> int:
    args = parse_args()
    info = wait_until_idle(
        args.gpu,
        args.max_used_mb,
        args.max_utilization,
        args.wait_timeout,
        args.poll_interval,
    )
    print(
        f"GPU {info.index} is idle: bus={info.bus_id}, "
        f"memory={info.memory_used_mb}/{info.memory_total_mb} MiB, utilization={info.utilization}%.",
        file=sys.stderr,
        flush=True,
    )
    guard_tensor = allocate_guard_tensor(args.gpu, args.hold_mb)
    try:
        if args.command:
            return run_command(args.command)
        return wait_forever()
    finally:
        del guard_tensor
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print("Released GPU guard.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
