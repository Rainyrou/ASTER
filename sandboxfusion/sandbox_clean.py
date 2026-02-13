#!/usr/bin/env python3
import os
import re
import signal
import time
from typing import Iterable

import psutil


CHECK_INTERVAL_SEC = int(os.getenv("CLEAN_INTERVAL_SEC", "60"))
MAX_AGE_SEC = int(os.getenv("CLEAN_MAX_AGE_SEC", "300"))  # 5 min


TMP_PY_RE = re.compile(r"^/tmp/(?:tmp[^/]+/)*tmp[^/]+\.py$")


def is_tmp_python_process(proc: psutil.Process) -> bool:
    try:
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        cmdline = proc.cmdline()
        if not cmdline:
            return False
        # 进程应为 python，可兼容不同可执行名
        exe = os.path.basename(cmdline[0]).lower()
        if "python" not in exe:
            return False
        # 任一参数命中 /tmp 下以 tmp*.py 命名的脚本
        for arg in cmdline[1:]:
            if isinstance(arg, str) and arg.endswith(".py") and TMP_PY_RE.match(arg):
                return True
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def kill_with_children(proc: psutil.Process) -> None:
    # 优先按进程组强杀，兜底逐个 kill
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        pass
    try:
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def find_candidates(processes: Iterable[psutil.Process], now: float) -> list[psutil.Process]:
    candidates: list[psutil.Process] = []
    for p in processes:
        try:
            if not is_tmp_python_process(p):
                continue
            age = now - p.create_time()
            if age >= MAX_AGE_SEC:
                candidates.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return candidates


def main() -> None:
    while True:
        now = time.time()
        procs = list(psutil.process_iter(["pid", "name", "cmdline", "create_time", "status"]))
        victims = find_candidates(procs, now)
        for v in victims:
            try:
                cmd = " ".join(v.cmdline())
            except Exception:
                cmd = "<unknown>"
            print(f"[clean.py] killing pid={v.pid} cmd={cmd}")
            kill_with_children(v)
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()


