import os
import platform
import json
import time
import shutil
import argparse
import multiprocessing as mp
import datetime


def _read_proc_stat():
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] != "cpu":
            return None
        values = list(map(int, parts[1:]))
        # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
        while len(values) < 10:
            values.append(0)
        user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice = values[:10]
        idle_all = idle + iowait
        non_idle = user + nice + system + irq + softirq + steal
        total = idle_all + non_idle
        return {
            "idle": idle_all,
            "non_idle": non_idle,
            "total": total,
        }
    except Exception:
        return None


def cpu_usage_percent(interval: float = 0.5) -> float | None:
    """Estimate CPU usage over a short interval using /proc/stat. Returns None if unavailable."""
    s1 = _read_proc_stat()
    if not s1:
        return None
    time.sleep(max(0.05, interval))
    s2 = _read_proc_stat()
    if not s2:
        return None
    totald = s2["total"] - s1["total"]
    idled = s2["idle"] - s1["idle"]
    if totald <= 0:
        return None
    usage = (totald - idled) / totald * 100.0
    return round(usage, 2)


def load_average():
    try:
        l1, l5, l15 = os.getloadavg()
        return {"1min": l1, "5min": l5, "15min": l15}
    except Exception:
        return None


def memory_info():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                val = v.strip().split()[0]
                info[k] = int(val)  # kB
    except Exception:
        return None

    def kb(x):
        return x

    def mb(x):
        return round(x / 1024.0, 2)

    def gb(x):
        return round(x / (1024.0 * 1024.0), 2)

    total_kb = kb(info.get("MemTotal", 0))
    free_kb = kb(info.get("MemFree", 0))
    avail_kb = kb(info.get("MemAvailable", 0))
    buffers_kb = kb(info.get("Buffers", 0))
    cached_kb = kb(info.get("Cached", 0))
    swap_total_kb = kb(info.get("SwapTotal", 0))
    swap_free_kb = kb(info.get("SwapFree", 0))

    used_kb = max(0, total_kb - avail_kb) if avail_kb else max(0, total_kb - free_kb - buffers_kb - cached_kb)
    return {
        "total_gb": gb(total_kb),
        "used_gb": gb(used_kb),
        "available_gb": gb(avail_kb),
        "free_gb": gb(free_kb),
        "buffers_gb": gb(buffers_kb),
        "cached_gb": gb(cached_kb),
        "swap_total_gb": gb(swap_total_kb),
        "swap_free_gb": gb(swap_free_kb),
    }


def disk_usage(paths: list[str] | None = None):
    if not paths:
        paths = ["/", os.getcwd()]
    seen = set()
    results = []
    for p in paths:
        try:
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            total, used, free = shutil.disk_usage(rp)
            results.append({
                "path": rp,
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round((used / total) * 100.0, 2) if total else None,
            })
        except Exception as e:
            results.append({"path": p, "error": str(e)})
    return results


def collect_info(sample_cpu: bool = True, cpu_interval: float = 0.5):
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": mp.cpu_count(),
        "load_avg": load_average(),
        "cpu_usage_percent": cpu_usage_percent(cpu_interval) if sample_cpu else None,
        "memory": memory_info(),
        "disks": disk_usage(),
        "env": {
            "SANDBOX_MAX_WORKERS": os.getenv("SANDBOX_MAX_WORKERS"),
            "SANDBOX_CPU_SECONDS": os.getenv("SANDBOX_CPU_SECONDS"),
            "SANDBOX_MEMORY_MB": os.getenv("SANDBOX_MEMORY_MB"),
            "SANDBOX_PROCESS_TIMEOUT": os.getenv("SANDBOX_PROCESS_TIMEOUT"),
        },
    }
    return info


def _print_human(data, cpu_interval: float):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Time       : {now}")
    print(f"Platform   : {data['platform']}")
    print(f"Python     : {data['python']}")
    print(f"CPU Count  : {data['cpu_count']}")
    if data["load_avg"]:
        la = data["load_avg"]
        print(f"Load Avg   : {la['1min']:.2f}, {la['5min']:.2f}, {la['15min']:.2f}")
    if data["cpu_usage_percent"] is not None:
        print(f"CPU Usage  : {data['cpu_usage_percent']:.2f}% (over {cpu_interval}s)")
    if data["memory"]:
        m = data["memory"]
        print(f"Memory     : total {m['total_gb']} GB, used {m['used_gb']} GB, avail {m['available_gb']} GB, free {m['free_gb']} GB")
        print(f"Swap       : total {m['swap_total_gb']} GB, free {m['swap_free_gb']} GB")
    print("Disks:")
    for d in data["disks"]:
        if "error" in d:
            print(f"  - {d['path']}: error: {d['error']}")
        else:
            print(f"  - {d['path']}: total {d['total_gb']} GB, used {d['used_gb']} GB, free {d['free_gb']} GB ({d['usage_percent']}%)")
    print("Env:")
    for k, v in data["env"].items():
        print(f"  - {k}={v}")


def main():
    # python3 ./aster/sandbox/sys_check.py --watch
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cpu-sample", action="store_true", help="Do not sample CPU usage (instant)")
    ap.add_argument("--cpu-interval", type=float, default=0.5, help="CPU sampling interval seconds")
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    ap.add_argument("--watch", action="store_true", help="Refresh continuously")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh interval seconds in watch mode")
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                data = collect_info(sample_cpu=(not args.no_cpu_sample), cpu_interval=args.cpu_interval)
                # 清屏重绘
                print("\033[2J\033[H", end="")
                if args.json:
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    _print_human(data, args.cpu_interval)
                # 采样周期与刷新周期分离：CPU 采样在 collect_info 内部
                time.sleep(max(0.1, args.interval))
        except KeyboardInterrupt:
            return
    else:
        data = collect_info(sample_cpu=(not args.no_cpu_sample), cpu_interval=args.cpu_interval)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            _print_human(data, args.cpu_interval)


if __name__ == "__main__":
    main()


