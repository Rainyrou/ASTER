# trace_event_exporter.py
import json
import os
import threading
import time
from typing import Dict, List, Optional


class TraceEventExporter:
    """
    Chrome Trace Event / Perfetto 导出器（支持 raw & aligned 两份文件）：
    - 每个 request_id 对应一个 tid（一行一个 request）
    - raw：所有事件共享同一时间原点（规范友好，真实时间轴）
    - aligned：每个 request 内部相对时间不变，但整体左对齐到本 request 的最早事件（仅用于形状对比）
    - 不包含 summary 轨道（按你的要求去掉）
    """

    def __init__(
        self,
        out_dir: str,
        filename_raw: str = "trace.perfetto.json",
        filename_aligned: str = "trace_aligned.perfetto.json",
        pid: int = 1,
    ):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.out_path_raw = os.path.join(out_dir, filename_raw)
        self.out_path_aligned = os.path.join(out_dir, filename_aligned)
        self.pid = pid

        self._lock = threading.RLock()

        # 事件缓存（存 raw 时间戳；aligned 在 dump 时再按 req_min 归零）
        self._events: List[Dict] = []

        # request_id -> tid（轨道 id）
        self._tid_map: Dict[str, int] = {}
        self._next_tid = 100

        # 是否已写 thread_name meta
        self._thread_named: Dict[int, bool] = {}

        # 全局最早绝对时间（raw 的统一原点）
        self._t0_global_abs: Optional[float] = None

        # 每个 request 的最早/最晚（raw ts，微秒）
        self._req_min_ts: Dict[str, int] = {}
        self._req_max_ts: Dict[str, int] = {}

    # ---- 时间换算：把 monotonic() 换成以“全局最早事件”为 0 的微秒 ----
    def _to_us_raw(self, t_abs: float) -> int:
        if self._t0_global_abs is None:
            self._t0_global_abs = t_abs
        return int((t_abs - self._t0_global_abs) * 1_000_000)

    # ---- tid 分配 & 命名 ----
    def _ensure_tid(self, request_id: str) -> int:
        tid = self._tid_map.get(request_id)
        if tid is None:
            tid = self._next_tid
            self._next_tid += 1
            self._tid_map[request_id] = tid
        if not self._thread_named.get(tid):
            # thread_name 元事件（无 ts）
            self._events.append({
                "ph": "M", "name": "thread_name",
                "pid": self.pid, "tid": tid,
                "args": {"name": f"req_{request_id}"}
            })
            self._thread_named[tid] = True
        return tid

    # ---- 事件写入（统一先写 raw ts；aligned 在 dump 时再偏移） ----
    def on_span_open(
        self,
        request_id: str,
        name: str,
        ts_monotonic: Optional[float] = None,
        cat: Optional[str] = None,
    ):
        with self._lock:
            t_abs = ts_monotonic if ts_monotonic is not None else time.monotonic()
            ts_raw = self._to_us_raw(t_abs)
            tid = self._ensure_tid(request_id)
            evt = {
                "ph": "B",
                "name": name,
                "cat": cat or "agent",
                "pid": self.pid,
                "tid": tid,
                "ts": ts_raw,
                "args": {}
            }
            self._events.append(evt)
            # 维护 per-request 最早/最晚
            self._req_min_ts[request_id] = ts_raw if request_id not in self._req_min_ts else min(self._req_min_ts[request_id], ts_raw)
            self._req_max_ts[request_id] = ts_raw if request_id not in self._req_max_ts else max(self._req_max_ts[request_id], ts_raw)

    def on_span_close(
        self,
        request_id: str,
        name: str,
        ts_monotonic: Optional[float] = None,
        cat: Optional[str] = None,
    ):
        with self._lock:
            t_abs = ts_monotonic if ts_monotonic is not None else time.monotonic()
            ts_raw = self._to_us_raw(t_abs)
            tid = self._ensure_tid(request_id)
            evt = {
                "ph": "E",
                "name": name,
                "cat": cat or "agent",
                "pid": self.pid,
                "tid": tid,
                "ts": ts_raw,
                "args": {}
            }
            self._events.append(evt)
            # 维护 per-request 最早/最晚
            self._req_min_ts[request_id] = ts_raw if request_id not in self._req_min_ts else min(self._req_min_ts[request_id], ts_raw)
            self._req_max_ts[request_id] = ts_raw if request_id not in self._req_max_ts else max(self._req_max_ts[request_id], ts_raw)

    # ---- dump：生成 raw 与 aligned 两份 ----
    def dump(self):
        with self._lock:
            # 1) raw：保持统一原点（规范友好）
            obj_raw = {
                "displayTimeUnit": "ms",
                "traceEvents": self._events
            }
            with open(self.out_path_raw, "w", encoding="utf-8") as f:
                json.dump(obj_raw, f, ensure_ascii=False)

            # 2) aligned：对每条 request 做 ts -= req_min_ts[req]
            if self._req_min_ts:
                events_aligned: List[Dict] = []
                for e in self._events:
                    # M 元事件没有 ts，照抄
                    if e.get("ph") == "M" or "ts" not in e:
                        events_aligned.append(dict(e))
                        continue
                    # 普通事件：按 request_id 归零
                    tid = e["tid"]
                    # 反查 request_id（O(1) 构造一个反向表）
                    # 由于 tid 稳定，预先建反向映射最简单
                    # 这里临时建一次也可以，事件量不大影响不大
                    # 构造一次反向映射
                # 先建反向映射
                tid_to_req = {tid: req for req, tid in self._tid_map.items()}
                for e in self._events:
                    if e.get("ph") == "M" or "ts" not in e:
                        events_aligned.append(dict(e))
                        continue
                    req = tid_to_req.get(e["tid"])
                    if req is None:
                        events_aligned.append(dict(e))
                        continue
                    e2 = dict(e)
                    e2["ts"] = e["ts"] - self._req_min_ts.get(req, 0)
                    events_aligned.append(e2)
            else:
                events_aligned = list(self._events)

            obj_aligned = {
                "displayTimeUnit": "ms",
                "traceEvents": events_aligned
            }
            with open(self.out_path_aligned, "w", encoding="utf-8") as f:
                json.dump(obj_aligned, f, ensure_ascii=False)

        return self.out_path_raw, self.out_path_aligned
