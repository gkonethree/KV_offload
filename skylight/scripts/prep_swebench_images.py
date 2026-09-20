#!/usr/bin/env python3
"""Pre-pull SWE-bench docker images for a subset of instance IDs.

Maps each ``repo__repo-issue`` id to the image mini-swe-agent uses::

    docker.io/swebench/sweb.eval.x86_64.{repo}_1776_{repo-issue}:latest

Usage::

    python scripts/prep_swebench_images.py bench-resources/swebench-verified-500.txt
    python scripts/prep_swebench_images.py --instances bench-resources/swebench-verified-200.txt --workers 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def instance_to_image(instance_id: str) -> str:
    slug = instance_id.strip().replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{slug}:latest"


def _read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _pull_one(image: str, retries: int, docker_exe: str) -> tuple[str, bool, str]:
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(
                [docker_exe, "pull", image],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0:
                return image, True, ""
            last_err = (r.stderr or r.stdout or "").strip()
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except OSError as exc:
            last_err = str(exc)
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    return image, False, last_err


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("instances", type=Path, nargs="?", help="one instance_id per line")
    p.add_argument("--instances", dest="instances_flag", type=Path, help="alias for positional")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=2, help="parallel docker pulls")
    p.add_argument(
        "--docker", default="docker",
        help="docker executable (or MSWEA_DOCKER_EXECUTABLE value)",
    )
    args = p.parse_args(argv)
    inst_path = args.instances_flag or args.instances
    if inst_path is None:
        p.error("provide an instances file")
    if not inst_path.exists():
        print(f"instances file not found: {inst_path}", file=sys.stderr)
        return 1

    ids = _read_ids(inst_path)
    images = sorted({instance_to_image(i) for i in ids})
    print(f"pulling {len(images)} unique images for {len(ids)} instances")

    ok, fail = 0, 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(_pull_one, img, args.retries, args.docker): img
            for img in images
        }
        for fut in as_completed(futs):
            image, success, err = fut.result()
            if success:
                ok += 1
                print(f"OK  {image}")
            else:
                fail += 1
                failures.append((image, err))
                print(f"FAIL {image}: {err[:200]}", file=sys.stderr)

    print(f"done: {ok} ok, {fail} failed")
    if failures:
        log = inst_path.parent / "prep_swebench_images_failures.txt"
        log.write_text("\n".join(f"{img}\t{err}" for img, err in failures) + "\n")
        print(f"failure log: {log}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
