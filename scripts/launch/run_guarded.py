#!/usr/bin/env python3
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(
    os.environ.get("MINIMIND_TRAIN_LOCK", "/tmp/minimind-lab-training.lock")
)
EXPECTED_GPU_COUNT = int(os.environ.get("EXPECTED_GPU_COUNT", "8"))


def fail(message: str) -> None:
    raise SystemExit(message)


def gpu_processes() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one guarded MiniMind training process."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Diagnostic only; record the exception in the experiment report.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a training command is required after --")

    if not args.skip_preflight:
        subprocess.run(
            [str(ROOT_DIR / "scripts/launch/preflight_l20.py")],
            check=True,
        )

    gpu_count = len(
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip().splitlines()
    )
    if gpu_count != EXPECTED_GPU_COUNT:
        fail(f"expected {EXPECTED_GPU_COUNT} GPUs, found {gpu_count}")

    existing = gpu_processes()
    if existing:
        print(existing, file=sys.stderr)
        fail("GPU compute processes already exist")

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail(f"another MiniMind run holds lock {LOCK_PATH}")

    print(f"launch_command={shlex.join(command)}", flush=True)
    if args.dry_run:
        print("dry_run=pass")
        return

    child: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            print(
                f"forwarding_signal={signal.Signals(signum).name} "
                f"child_pid={child.pid}",
                file=sys.stderr,
                flush=True,
            )
            os.killpg(child.pid, signum)

    for signal_name in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_name, forward)

    child = subprocess.Popen(command, start_new_session=True)
    while True:
        try:
            child_rc = child.wait()
            break
        except InterruptedError:
            continue

    if received_signal is not None:
        print(
            f"run_status=interrupted "
            f"signal={signal.Signals(received_signal).name} child_rc={child_rc}",
            file=sys.stderr,
        )
        raise SystemExit(128 + received_signal)

    print(f"run_status=finished child_rc={child_rc}")
    raise SystemExit(child_rc)


if __name__ == "__main__":
    main()
