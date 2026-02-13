# speedscope_exporter.py
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple


class SpeedscopeExporter:
    """
    单 profile + 多 request 一行一个：
    - 整个 trace 只导出一个 profile
    - 每个 request 占独立一行（顶层 frame）
    - 子 span 命名为 "request <id>/<turn_0>/<llm_generate>" 这种层级
    """

    def __init__(
        self,
        out_dir: str,
        filename: str = "trace.speedscope.json",
        default_category: str = "agent",
        unit: str = "microseconds",
    ):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.out_path = os.path.join(out_dir, filename)
        self.default_category = default_category
        self.unit = unit

        self._lock = threading.RLock()

        # 全局共享 frame 表
        self._frames: List[Dict] = []
        self._frame_index: Dict[Tuple[str, str], int] = {}

        # 单 profile 的事件序列
        self._events: List[Dict] = []

        # 时间对齐
        self._t0_abs: Optional[float] = None
        self._end_value: float = 0.0

    # ---- 时间换算 ---------------------------------------------------------
    def _ts_to_value(self, t_abs: float) -> float:
        if self._t0_abs is None:
            self._t0_abs = t_abs
        delta = t_abs - self._t0_abs
        if self.unit == "microseconds":
            return delta * 1e6
        elif self.unit == "milliseconds":
            return delta * 1e3
        elif self.unit == "nanoseconds":
            return delta * 1e9
        elif self.unit == "seconds":
            return delta
        else:
            return delta

    def _ensure_end_value(self, v: float):
        if v > self._end_value:
            self._end_value = v

    # ---- frame & event ---------------------------------------------------
    def _get_frame_index(self, name: str, category: Optional[str]) -> int:
        key = (name, category or self.default_category)
        idx = self._frame_index.get(key)
        if idx is not None:
            return idx
        idx = len(self._frames)
        self._frame_index[key] = idx
        self._frames.append({"name": name, "category": key[1]})
        return idx

    def on_span_open(
        self,
        request_id: str,
        name: str,
        ts_monotonic: Optional[float] = None,
        category: Optional[str] = None,
    ):
        """记录 span 开始"""
        with self._lock:
            t_abs = ts_monotonic if ts_monotonic is not None else time.monotonic()
            at = self._ts_to_value(t_abs)
            # 带上 request_id 作为前缀
            frame_name = f"req_{request_id}/{name}"
            frame = self._get_frame_index(frame_name, category)
            self._events.append({"type": "O", "at": at, "frame": frame})
            self._ensure_end_value(at)

    def on_span_close(
        self,
        request_id: str,
        name: str,
        ts_monotonic: Optional[float] = None,
        category: Optional[str] = None,
    ):
        """记录 span 结束"""
        with self._lock:
            t_abs = ts_monotonic if ts_monotonic is not None else time.monotonic()
            at = self._ts_to_value(t_abs)
            frame_name = f"req_{request_id}/{name}"
            frame = self._get_frame_index(frame_name, category)
            self._events.append({"type": "C", "at": at, "frame": frame})
            self._ensure_end_value(at)

    # ---- dump ------------------------------------------------------------
    def dump(self, name: str = "AgentLoop Trace"):
        with self._lock:
            data = {
                "$schema": "https://www.speedscope.app/file-format-schema.json",
                "activeProfileIndex": 0,
                "name": name,
                "shared": {"frames": self._frames},
                "profiles": [
                    {
                        "name": "all_requests",
                        "type": "evented",
                        "unit": self.unit,
                        "startValue": 0.0,
                        "endValue": self._end_value,
                        "events": self._events,
                    }
                ],
            }
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        return self.out_path
