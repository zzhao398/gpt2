#!/usr/bin/env python3

import argparse
import csv
import itertools
import os
import shutil
import subprocess
import time

U_VALS         = [8, 16, 32]
TILE_WIDTH_VALS = [8, 16, 32]

SMEM_LIMIT   = 49152
THREAD_LIMIT = 1024
FLOAT_BYTES  = 4


def is_valid(U, TILE_WIDTH):
    BLOCK_SIZE = U * TILE_WIDTH
    if BLOCK_SIZE > THREAD_LIMIT or BLOCK_SIZE < 32:
        return False
    smem = (BLOCK_SIZE * TILE_WIDTH + TILE_WIDTH * U) * FLOAT_BYTES
    if smem > SMEM_LIMIT:
        return False
    return True


def run_config(cfg, repo_root, template, kernel_dst, timeout):
    with open(kernel_dst, "w") as f:
        f.write(template.format(**cfg))

    subprocess.run(["make", "clean"], cwd=repo_root, capture_output=True)
    build = subprocess.run(
        ["make", "test_gpt2_kernels"],
        cwd=repo_root, capture_output=True, text=True, timeout=timeout
    )
    if build.returncode != 0:
        return None, "compile_error", build.stderr[-300:]

    t0 = time.perf_counter()
    run = subprocess.run(
        ["./test_gpt2_kernels"],
        cwd=repo_root, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.perf_counter() - t0

    if run.returncode != 0:
        return elapsed, "runtime_error", run.stderr[-200:]

    status = "ok" if "All kernel tests done" in run.stdout else "wrong_answer"
    return elapsed, status, ""


def main():
    parser = argparse.ArgumentParser(description="Sweep matmul tile configurations")
    parser.add_argument("--out", default="kernels_op_7/sweep_results.csv")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    repo_root     = os.path.abspath(os.path.join(script_dir, ".."))
    template_path = os.path.join(script_dir, "matmul.cuh")
    kernel_dst    = os.path.join(repo_root, "kernels", "matmul.cuh")
    kernel_bak    = kernel_dst + ".sweep_bak"
    out_csv       = os.path.join(repo_root, args.out)

    with open(template_path) as f:
        template = f.read()

    shutil.copy2(kernel_dst, kernel_bak)

    configs = [
        {"U": u, "TILE_WIDTH": tw, "BLOCK_SIZE": u * tw}
        for u, tw in itertools.product(U_VALS, TILE_WIDTH_VALS)
        if is_valid(u, tw)
    ]
    print(f"Valid configurations to test: {len(configs)}", flush=True)

    rows = []
    try:
        for i, cfg in enumerate(configs):
            label = "U={U} TILE_WIDTH={TILE_WIDTH} BLOCK_SIZE={BLOCK_SIZE}".format(**cfg)
            print(f"[{i+1}/{len(configs)}] {label} ...", end=" ", flush=True)

            try:
                elapsed, status, err = run_config(
                    cfg, repo_root, template, kernel_dst, args.timeout
                )
            except subprocess.TimeoutExpired:
                elapsed, status, err = None, "timeout", ""
            except Exception as e:
                elapsed, status, err = None, "exception", str(e)

            smem = (cfg["BLOCK_SIZE"] * cfg["TILE_WIDTH"] + cfg["TILE_WIDTH"] * cfg["U"]) * FLOAT_BYTES
            rows.append({
                **cfg,
                "smem_bytes": smem,
                "elapsed_s":  f"{elapsed:.4f}" if elapsed is not None else "",
                "status":     status,
                "note":       err,
            })
            print(f"{status}  {elapsed:.3f}s" if elapsed is not None else status, flush=True)

    finally:
        shutil.copy2(kernel_bak, kernel_dst)
        os.remove(kernel_bak)
        print("\nOriginal matmul.cuh restored.")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ["U", "TILE_WIDTH", "BLOCK_SIZE", "smem_bytes", "elapsed_s", "status", "note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: float(r["elapsed_s"]))
        print(
            f"\nBest: U={best['U']} TILE_WIDTH={best['TILE_WIDTH']} BLOCK_SIZE={best['BLOCK_SIZE']} "
            f"-> {best['elapsed_s']}s "
            f"({best['smem_bytes']} bytes smem)"
        )
    print(f"Results: {out_csv}")


if __name__ == "__main__":
    main()
