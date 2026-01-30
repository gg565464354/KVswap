#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
from multiprocessing import Process

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRED_PY = os.path.join(ROOT, "pred.py")

DEFAULT_TASKS = [
    "hotpotqa",
    "narrativeqa",
    "multifieldqa_en",
    "musique",
    "dureader",
    "gov_report",
    "samsum",
    "passage_retrieval_en",
    "lcc",
]

DEFAULT_TASKS_E = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pred.py default tasks with optional multi-GPU parallelism."
    )
    parser.add_argument("--model_name", required=True, help="Model name in config/model2path.json")
    parser.add_argument("--method", default="full", help="Method passed to pred.py")
    parser.add_argument("--task", default=None, help="Run a single task (overrides default task list)")
    parser.add_argument("--e", action="store_true", help="Use LongBench-E task list")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids (e.g., 0,1,2). Default: all")

    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--datalen", type=int, default=128 * 1024)
    parser.add_argument("--cache_dir", type=str, default="/root/.cache/datasets/THUDM___long_bench")
    parser.add_argument("--name", type=str, default=None)

    args, extra = parser.parse_known_args()
    return args, extra


def resolve_gpus(arg):
    if arg is None or arg == "":
        count = torch.cuda.device_count()
        if count <= 0:
            raise RuntimeError("No CUDA devices found. pred.py uses cuda:0, GPU is required.")
        return list(range(count))
    gpus = []
    for part in arg.split(","):
        part = part.strip()
        if part == "":
            continue
        gpus.append(int(part))
    if not gpus:
        raise RuntimeError("--gpus provided but parsed empty list")
    return gpus


def build_cmd(args, extra, task):
    cmd = [sys.executable, PRED_PY, "--model_name", args.model_name, "--method", args.method]
    if args.e:
        cmd.append("--e")
    if task is not None:
        cmd += ["--task", task]

    cmd += ["--num_examples", str(args.num_examples)]
    cmd += ["--batch_size", str(args.batch_size)]
    cmd += ["--datalen", str(args.datalen)]
    cmd += ["--cache_dir", args.cache_dir]
    if args.name is not None:
        cmd += ["--name", args.name]

    if extra:
        cmd += extra
    return cmd


def worker(gpu_id, tasks, args, extra):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for task in tasks:
        cmd = build_cmd(args, extra, task)
        print(f"[GPU {gpu_id}] Running task: {task}")
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Task {task} failed on GPU {gpu_id} with code {proc.returncode}")


def main():
    args, extra = parse_args()

    if not os.path.exists(PRED_PY):
        raise RuntimeError(f"pred.py not found at {PRED_PY}")

    if args.task:
        tasks = [args.task]
    else:
        tasks = DEFAULT_TASKS_E if args.e else DEFAULT_TASKS

    gpus = resolve_gpus(args.gpus)

    # Split tasks round-robin across GPUs
    buckets = [[] for _ in gpus]
    for idx, task in enumerate(tasks):
        buckets[idx % len(gpus)].append(task)

    procs = []
    for gpu_id, task_list in zip(gpus, buckets):
        if not task_list:
            continue
        p = Process(target=worker, args=(gpu_id, task_list, args, extra))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(p.exitcode)

    print("All tasks finished.")


if __name__ == "__main__":
    main()
