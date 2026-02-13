# timeline.py
import asyncio
import json
import os
import time
import atexit
import uuid
import contextvars
from datetime import datetime
from typing import Any, Optional, Dict

# Speedscope（单 profile，多 request 行）
try:
    from .speedscope import SpeedscopeExporter
except Exception:
    SpeedscopeExporter = None

# Trace Event / Perfetto（多轨道，每个 request 一行）
try:
    from .trace_event import TraceEventExporter
except Exception:
    TraceEventExporter = None

_request_id_var = contextvars.ContextVar("request_id", default=None)
_turn_var = contextvars.ContextVar("turn", default=None)

class Timeline:
    """异步 JSONL + Speedscope + TraceEvent 记录器"""
    def __init__(
        self,
        base_dir: str,
        worker_id: str,
        enabled: bool = True,
        queue_maxsize: int = 10000,
        encoding: str = "utf-8",
        # Speedscope
        enable_speedscope: bool = False,
        speedscope_filename: str = "trace.speedscope.json",
        speedscope_unit: str = "microseconds",
        # Trace Event / Perfetto
        enable_trace_event: bool = False,
        trace_event_filename: str = "trace.perfetto.json",
        use_step_subdir: bool = True,
    ):
        self.enabled = enabled
        self.base_dir = base_dir
        self.worker_id = worker_id
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._writer_task: Optional[asyncio.Task] = None
        self._closed = False
        self._fh_cache: Dict[str, Any] = {}
        self._encoding = encoding

        # Speedscope 单 profile，多 request
        self._speedscope = None
        if enable_speedscope and SpeedscopeExporter is not None:
            self._speedscope = SpeedscopeExporter(
                out_dir=base_dir,
                filename=speedscope_filename,
                unit=speedscope_unit,
            )

        # Trace Event / Perfetto 每个 request 一条轨道
        self._trace_event = None
        if enable_trace_event and TraceEventExporter is not None:
            self._trace_event = TraceEventExporter(
                out_dir=base_dir,
                filename_raw=trace_event_filename,
                filename_aligned=trace_event_filename.replace(".json", "_aligned.json"),
            )

        self._use_step_subdir = use_step_subdir

        atexit.register(self._sync_close)

    # 事件环 & 关闭
    def ensure_started(self):
        if not self.enabled:
            return
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(self._writer_loop())

    async def aclose(self):
        if self._closed or not self.enabled:
            # 兜底 dump
            try:
                if self._speedscope:
                    self._speedscope.dump(name=f"{self.worker_id} trace")
                if self._trace_event:
                    self._trace_event.dump()
            except Exception:
                pass
            return
        self._closed = True
        await self.queue.put(None)
        if self._writer_task:
            await self._writer_task
        try:
            if self._speedscope:
                self._speedscope.dump(name=f"{self.worker_id} trace")
            if self._trace_event:
                self._trace_event.dump()
        except Exception:
            pass

    def _sync_close(self):
        for fh in self._fh_cache.values():
            try:
                fh.close()
            except Exception:
                pass
        try:
            if self._speedscope:
                self._speedscope.dump(name=f"{self.worker_id} trace")
            if self._trace_event:
                self._trace_event.dump()
        except Exception:
            pass

    # JSONL 写入
    def _path_for(self, step: Optional[int]) -> str:
        # ✅ 如果明确关闭了 step 子目录，或没提供 step，就直接写到 base_dir
        if not self._use_step_subdir or step is None:
            path = self.base_dir
        else:
            step_dir = f"step_{-1 if step is None else step}"
            path = os.path.join(self.base_dir, step_dir)

        os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"{self.worker_id}.jsonl")
    
    async def _writer_loop(self):
        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                step = item.pop("_step", -1)
                path = self._path_for(step)
                fh = self._fh_cache.get(path)
                if fh is None:
                    fh = open(path, "a", buffering=1, encoding=self._encoding)
                    self._fh_cache[path] = fh
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                self.queue.task_done()
        finally:
            for fh in self._fh_cache.values():
                try:
                    fh.close()
                except Exception:
                    pass

    # 原子事件
    def log(self, name: str, *, step: Optional[int] = None, **fields: Any):
        if not self.enabled:
            return
        payload = {
            "ts": datetime.now().isoformat(),
            "name": name,
            "request_id": _request_id_var.get(),
            "turn": _turn_var.get(),
            **fields,
            "_step": step,
        }
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    # Span（异步上下文管理器）
    def span(self, name: str, *, step: Optional[int] = None,
             category: Optional[str] = None, **fields: Any):
        return _AsyncSpan(self, name, step=step, category=category, fields=fields)

    # 上下文变量
    def set_request(self, rid: str):
        return _VarToken(_request_id_var, rid)

    def set_turn(self, turn_idx: Optional[int]):
        return _VarToken(_turn_var, turn_idx)


class _AsyncSpan:
    """记录 start/end（JSONL + speedscope + traceEvent）的异步 span"""
    def __init__(self, tl: Timeline, name: str, *, step: Optional[int],
                 category: Optional[str], fields: Dict[str, Any]):
        self.tl = tl
        self.name = name
        self.step = step
        self.category = category
        self.fields = fields
        self._t0 = 0.0
        self._span_id = uuid.uuid4().hex[:12]
        self._t0_mono = 0.0

    async def __aenter__(self):
        self._t0 = time.perf_counter()
        self._t0_mono = time.monotonic()
        self.tl.log(f"{self.name}.start", step=self.step,
                    span_id=self._span_id, **self.fields)

        # Speedscope：单 profile + req 前缀
        if self.tl._speedscope:
            try:
                req_id = _request_id_var.get() or "no_req"
                self.tl._speedscope.on_span_open(
                    req_id, self.name, ts_monotonic=self._t0_mono, category=self.category
                )
            except Exception:
                pass

        # Trace Event：每个 req 一条轨道
        if self.tl._trace_event:
            try:
                req_id = _request_id_var.get() or "no_req"
                self.tl._trace_event.on_span_open(
                    req_id, self.name, ts_monotonic=self._t0_mono, cat=self.category
                )
            except Exception:
                pass

        return self

    async def __aexit__(self, et, ev, tb):
        dur = time.perf_counter() - self._t0
        status = "ok" if et is None else "error"
        self.tl.log(
            f"{self.name}.end",
            step=self.step,
            span_id=self._span_id,
            status=status,
            duration_sec=dur,
            error=str(ev) if ev else None,
        )

        # Speedscope
        if self.tl._speedscope:
            try:
                req_id = _request_id_var.get() or "no_req"
                self.tl._speedscope.on_span_close(
                    req_id, self.name, ts_monotonic=time.monotonic(), category=self.category
                )
            except Exception:
                pass

        # Trace Event
        if self.tl._trace_event:
            try:
                req_id = _request_id_var.get() or "no_req"
                self.tl._trace_event.on_span_close(
                    req_id, self.name, ts_monotonic=time.monotonic(), cat=self.category
                )
            except Exception:
                pass

        return False

    def event(self, subname: str, **fields: Any):
        self.tl.log(f"{self.name}.{subname}", step=self.step,
                    span_id=self._span_id, **fields)


class _VarToken:
    def __init__(self, var: contextvars.ContextVar, value):
        self.var = var
        self.token = var.set(value)

    def reset(self):
        try:
            self.var.reset(self.token)
        except Exception:
            pass
