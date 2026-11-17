# memory_timer.py

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Dict, List, Optional
import os
import time
import json
import tracemalloc


# ============================================================
# Optional process-memory support
# ============================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


# ============================================================
# Basic formatting
# ============================================================

def bytes_to_mb(n_bytes):
    return float(n_bytes) / (1024.0 * 1024.0)


def mb_to_bytes(mb):
    return int(float(mb) * 1024.0 * 1024.0)


def format_seconds(seconds):
    seconds = float(seconds)

    if seconds < 1e-6:
        return "{:.3f} ns".format(seconds * 1e9)

    if seconds < 1e-3:
        return "{:.3f} µs".format(seconds * 1e6)

    if seconds < 1.0:
        return "{:.3f} ms".format(seconds * 1e3)

    if seconds < 60.0:
        return "{:.3f} s".format(seconds)

    minutes = int(seconds // 60.0)
    rem = seconds - 60.0 * minutes

    if minutes < 60:
        return "{} min {:.2f} s".format(minutes, rem)

    hours = int(minutes // 60)
    minutes = minutes - 60 * hours

    return "{} h {} min {:.2f} s".format(hours, minutes, rem)


def format_mb(mb):
    mb = float(mb)

    if mb < 1.0:
        return "{:.3f} MB".format(mb)

    if mb < 1024.0:
        return "{:.2f} MB".format(mb)

    return "{:.2f} GB".format(mb / 1024.0)


def format_bytes(n_bytes):
    return format_mb(bytes_to_mb(n_bytes))


# ============================================================
# Process memory
# ============================================================

def get_process_memory_bytes():
    """
    Return current process RSS memory in bytes.

    On Windows, Linux, and macOS, psutil is the best route.
    If psutil is unavailable, returns None.
    """
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss)

    return None


def get_process_memory_mb():
    value = get_process_memory_bytes()

    if value is None:
        return None

    return bytes_to_mb(value)


def ensure_tracemalloc_started():
    if not tracemalloc.is_tracing():
        tracemalloc.start()


def get_tracemalloc_memory_bytes():
    """
    Return current and peak Python-tracked memory.

    This is not the same as process RSS. NumPy arrays are often partly outside
    what tracemalloc sees. Use RSS if psutil is available.
    """
    if not tracemalloc.is_tracing():
        return None, None

    current, peak = tracemalloc.get_traced_memory()

    return int(current), int(peak)


# ============================================================
# Data containers
# ============================================================

@dataclass
class MemorySnapshot:
    label: str
    timestamp_s: float
    rss_mb: Optional[float]
    traced_current_mb: Optional[float]
    traced_peak_mb: Optional[float]


@dataclass
class TimingRecord:
    name: str

    start_time_s: float
    end_time_s: float
    elapsed_s: float

    rss_start_mb: Optional[float]
    rss_end_mb: Optional[float]
    rss_delta_mb: Optional[float]

    traced_start_mb: Optional[float]
    traced_end_mb: Optional[float]
    traced_delta_mb: Optional[float]

    traced_peak_start_mb: Optional[float]
    traced_peak_end_mb: Optional[float]
    traced_peak_delta_mb: Optional[float]

    metadata: Dict = field(default_factory=dict)


# ============================================================
# Timer context manager
# ============================================================

class TimerBlock(object):
    """
    Simple timing + memory context manager.

    Example
    -------
    with TimerBlock("preprocessing") as t:
        do_work()

    print(t.record.elapsed_s)
    """

    def __init__(self, name, profiler=None, metadata=None, start_tracemalloc=True, verbose=False):
        self.name = str(name)
        self.profiler = profiler
        self.metadata = {} if metadata is None else dict(metadata)
        self.start_tracemalloc = bool(start_tracemalloc)
        self.verbose = bool(verbose)

        self.start_time_s = None
        self.end_time_s = None

        self.rss_start_mb = None
        self.rss_end_mb = None

        self.traced_start_mb = None
        self.traced_end_mb = None

        self.traced_peak_start_mb = None
        self.traced_peak_end_mb = None

        self.record = None

    def __enter__(self):
        if self.start_tracemalloc:
            ensure_tracemalloc_started()

        self.start_time_s = time.perf_counter()

        self.rss_start_mb = get_process_memory_mb()

        traced_current, traced_peak = get_tracemalloc_memory_bytes()

        self.traced_start_mb = None if traced_current is None else bytes_to_mb(traced_current)
        self.traced_peak_start_mb = None if traced_peak is None else bytes_to_mb(traced_peak)

        if self.verbose:
            print("[START] {}".format(self.name))

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time_s = time.perf_counter()

        self.rss_end_mb = get_process_memory_mb()

        traced_current, traced_peak = get_tracemalloc_memory_bytes()

        self.traced_end_mb = None if traced_current is None else bytes_to_mb(traced_current)
        self.traced_peak_end_mb = None if traced_peak is None else bytes_to_mb(traced_peak)

        elapsed_s = self.end_time_s - self.start_time_s

        rss_delta_mb = _delta_optional(self.rss_end_mb, self.rss_start_mb)
        traced_delta_mb = _delta_optional(self.traced_end_mb, self.traced_start_mb)
        traced_peak_delta_mb = _delta_optional(self.traced_peak_end_mb, self.traced_peak_start_mb)

        self.record = TimingRecord(
            name=self.name,
            start_time_s=float(self.start_time_s),
            end_time_s=float(self.end_time_s),
            elapsed_s=float(elapsed_s),

            rss_start_mb=self.rss_start_mb,
            rss_end_mb=self.rss_end_mb,
            rss_delta_mb=rss_delta_mb,

            traced_start_mb=self.traced_start_mb,
            traced_end_mb=self.traced_end_mb,
            traced_delta_mb=traced_delta_mb,

            traced_peak_start_mb=self.traced_peak_start_mb,
            traced_peak_end_mb=self.traced_peak_end_mb,
            traced_peak_delta_mb=traced_peak_delta_mb,

            metadata=self.metadata,
        )

        if self.profiler is not None:
            self.profiler.add_record(self.record)

        if self.verbose:
            print("[END] {} | {}".format(self.name, format_seconds(elapsed_s)))

        # Do not suppress exceptions.
        return False


def _delta_optional(a, b):
    if a is None or b is None:
        return None

    return float(a) - float(b)


# ============================================================
# Pipeline profiler
# ============================================================

class PipelineProfiler(object):
    """
    Collect timing/memory records for a full pipeline.

    Example
    -------
    profiler = PipelineProfiler("lobe pipeline")

    with profiler.block("generate scene"):
        ...

    with profiler.block("preprocessing"):
        ...

    profiler.print_summary()
    """

    def __init__(self, name="pipeline", start_tracemalloc=True):
        self.name = str(name)
        self.records = []
        self.snapshots = []
        self.start_time_s = time.perf_counter()
        self.end_time_s = None

        self.start_tracemalloc = bool(start_tracemalloc)

        if self.start_tracemalloc:
            ensure_tracemalloc_started()

        self.add_snapshot("start")

    def block(self, name, metadata=None, verbose=False):
        return TimerBlock(
            name=name,
            profiler=self,
            metadata=metadata,
            start_tracemalloc=self.start_tracemalloc,
            verbose=verbose,
        )

    def add_record(self, record):
        self.records.append(record)

    def add_snapshot(self, label):
        rss_mb = get_process_memory_mb()

        traced_current, traced_peak = get_tracemalloc_memory_bytes()

        snapshot = MemorySnapshot(
            label=str(label),
            timestamp_s=float(time.perf_counter()),
            rss_mb=rss_mb,
            traced_current_mb=None if traced_current is None else bytes_to_mb(traced_current),
            traced_peak_mb=None if traced_peak is None else bytes_to_mb(traced_peak),
        )

        self.snapshots.append(snapshot)

        return snapshot

    def finish(self):
        self.end_time_s = time.perf_counter()
        self.add_snapshot("finish")

    def total_elapsed_s(self):
        if self.end_time_s is None:
            return float(time.perf_counter() - self.start_time_s)

        return float(self.end_time_s - self.start_time_s)

    def summary_dict(self):
        total_elapsed = self.total_elapsed_s()

        records = []

        for r in self.records:
            records.append(timing_record_to_dict(r))

        snapshots = []

        for s in self.snapshots:
            snapshots.append(memory_snapshot_to_dict(s))

        return {
            "name": self.name,
            "total_elapsed_s": float(total_elapsed),
            "total_elapsed_pretty": format_seconds(total_elapsed),
            "rss_available": bool(PSUTIL_AVAILABLE),
            "tracemalloc_available": bool(tracemalloc.is_tracing()),
            "records": records,
            "snapshots": snapshots,
        }

    def print_summary(self, include_metadata=False):
        print()
        print("=" * 110)
        print("PIPELINE PROFILER SUMMARY")
        print("=" * 110)
        print("name:", self.name)
        print("total elapsed:", format_seconds(self.total_elapsed_s()))
        print("rss/process memory available:", PSUTIL_AVAILABLE)
        print("tracemalloc active:", tracemalloc.is_tracing())
        print("-" * 110)

        headers = [
            "step",
            "time",
            "rss start",
            "rss end",
            "rss Δ",
            "trace start",
            "trace end",
            "trace Δ",
            "peak Δ",
        ]

        widths = [28, 13, 13, 13, 13, 13, 13, 13, 13]

        print(_format_row(headers, widths))
        print("-" * 110)

        for r in self.records:
            row = [
                r.name,
                format_seconds(r.elapsed_s),
                _fmt_optional_mb(r.rss_start_mb),
                _fmt_optional_mb(r.rss_end_mb),
                _fmt_optional_mb_delta(r.rss_delta_mb),
                _fmt_optional_mb(r.traced_start_mb),
                _fmt_optional_mb(r.traced_end_mb),
                _fmt_optional_mb_delta(r.traced_delta_mb),
                _fmt_optional_mb_delta(r.traced_peak_delta_mb),
            ]

            print(_format_row(row, widths))

            if include_metadata and r.metadata:
                print("    metadata:", r.metadata)

        print("-" * 110)

        current_rss = get_process_memory_mb()

        if current_rss is not None:
            print("current RSS:", format_mb(current_rss))

        traced_current, traced_peak = get_tracemalloc_memory_bytes()

        if traced_current is not None:
            print("current traced:", format_bytes(traced_current))
            print("peak traced:", format_bytes(traced_peak))

        print("=" * 110)
        print()

    def save_json(self, path):
        path = str(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary_dict(), f, indent=2)

        return path


def _format_row(values, widths):
    cells = []

    for value, width in zip(values, widths):
        text = str(value)

        if len(text) > width:
            text = text[:max(0, width - 1)] + "…"

        cells.append(text.ljust(width))

    return " ".join(cells)


def _fmt_optional_mb(value):
    if value is None:
        return "n/a"

    return format_mb(value)


def _fmt_optional_mb_delta(value):
    if value is None:
        return "n/a"

    sign = "+" if value >= 0.0 else ""

    return sign + format_mb(value)


def timing_record_to_dict(record):
    return {
        "name": record.name,
        "start_time_s": float(record.start_time_s),
        "end_time_s": float(record.end_time_s),
        "elapsed_s": float(record.elapsed_s),
        "elapsed_pretty": format_seconds(record.elapsed_s),

        "rss_start_mb": _json_optional_float(record.rss_start_mb),
        "rss_end_mb": _json_optional_float(record.rss_end_mb),
        "rss_delta_mb": _json_optional_float(record.rss_delta_mb),

        "traced_start_mb": _json_optional_float(record.traced_start_mb),
        "traced_end_mb": _json_optional_float(record.traced_end_mb),
        "traced_delta_mb": _json_optional_float(record.traced_delta_mb),

        "traced_peak_start_mb": _json_optional_float(record.traced_peak_start_mb),
        "traced_peak_end_mb": _json_optional_float(record.traced_peak_end_mb),
        "traced_peak_delta_mb": _json_optional_float(record.traced_peak_delta_mb),

        "metadata": record.metadata,
    }


def memory_snapshot_to_dict(snapshot):
    return {
        "label": snapshot.label,
        "timestamp_s": float(snapshot.timestamp_s),
        "rss_mb": _json_optional_float(snapshot.rss_mb),
        "traced_current_mb": _json_optional_float(snapshot.traced_current_mb),
        "traced_peak_mb": _json_optional_float(snapshot.traced_peak_mb),
    }


def _json_optional_float(value):
    if value is None:
        return None

    return float(value)


# ============================================================
# Convenience functions
# ============================================================

@contextmanager
def timed_step(name, profiler=None, metadata=None, verbose=False):
    """
    Lightweight contextmanager wrapper.

    Example
    -------
    with timed_step("preprocessing", profiler):
        ...
    """
    block = TimerBlock(
        name=name,
        profiler=profiler,
        metadata=metadata,
        verbose=verbose,
    )

    with block:
        yield block


def print_current_memory(label="current memory"):
    rss = get_process_memory_mb()

    traced_current, traced_peak = get_tracemalloc_memory_bytes()

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    if rss is None:
        print("RSS/process memory: n/a")
    else:
        print("RSS/process memory:", format_mb(rss))

    if traced_current is None:
        print("tracemalloc: n/a")
    else:
        print("traced current:", format_bytes(traced_current))
        print("traced peak:", format_bytes(traced_peak))

    print("=" * 80)
    print()


def measure_array_memory_mb(*arrays):
    """
    Return total nbytes of NumPy-like arrays in MB.
    """
    total = 0

    for arr in arrays:
        if arr is None:
            continue

        if hasattr(arr, "nbytes"):
            total += int(arr.nbytes)

    return bytes_to_mb(total)


def describe_array_memory(name, *arrays):
    mb = measure_array_memory_mb(*arrays)

    return {
        "name": str(name),
        "memory_mb": float(mb),
        "memory_pretty": format_mb(mb),
    }


# ============================================================
# Small self-test
# ============================================================

if __name__ == "__main__":
    import numpy as np

    profiler = PipelineProfiler("memory_timer_selftest")

    with profiler.block("small array allocation"):
        a = np.zeros((1000, 1000), dtype=np.float64)
        a += 1.0

    with profiler.block("second array allocation"):
        b = np.ones((500, 500), dtype=np.float64)
        c = a[:500, :500] + b

    profiler.finish()
    profiler.print_summary(include_metadata=True)

    print(describe_array_memory("arrays", a, b, c))