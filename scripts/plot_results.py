#!/usr/bin/env python3
"""plot_results.py -- cross-GPU result plots and README for MT4G sample results.

Reads the MT4G result files, draws one set of comparison figures with the GPU
model on the x-axis, and writes ``sample_results/README.md``, which embeds every
figure and lists the underlying values.

Inputs (all discovered; no GPU model or configuration is hard-coded):
  * ``<results-dir>/*.json``  -- one MT4G JSON per GPU. Used for latency, cache
    size, cache line size and fetch granularity. The bandwidth fields in these
    files are ignored: they predate the current sweep benchmarks.
  * ``<results-dir>/**/*_BW_Grid.csv`` -- bandwidth sweep grids from the current
    benchmarks in ``src/benchmarks/bandwidth/``. One plot is drawn per direction and
    launch configuration (per-SM/CU levels: per thread count; device-wide levels:
    per block count x thread count), even before any measurement exists
    (template plots); grids added later fill them in.

Usage:
  scripts/plot_results.py
  scripts/plot_results.py --results-dir sample_results --outdir sample_results/plots

Requires matplotlib (see the plot venv notes in the main README).
"""
from __future__ import annotations

import argparse
import json
import io
import math
import os
import re
from collections import defaultdict
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from PIL import Image


# Same cut-off main.cpp uses (VALIDITY_THRESHOLD) to decide whether a measured
# size / line-size / fetch-granularity result is trustworthy.
VALIDITY_THRESHOLD = 0.5
DPI = 200
REPO_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 14.5,
        "axes.labelsize": 13.5,
        "legend.fontsize": 12.5,
        "legend.title_fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "0.88",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.8,
        "svg.hashsalt": "mt4g",
    }
)

TEXT_MUTED = "#6b6a66"


# --------------------------------------------------------------------------- #
# Memory hierarchy levels: one fixed colour per level, used by every figure.   #
# Eight hues; the two "sub-levels" (scalar L1, constant L1.5, L2 segment)      #
# reuse their family's hue with a hatch so no ninth hue is needed.             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Level:
    key: str
    label: str
    color: str
    hatch: Optional[str]
    vendors: frozenset
    json_node: tuple  # path below result["memory"]


BOTH = frozenset({"AMD", "NVIDIA"})
AMD = frozenset({"AMD"})
NVIDIA = frozenset({"NVIDIA"})

LEVELS: dict[str, Level] = {
    lv.key: lv
    for lv in [
        Level("shared", "Shared memory / LDS", "#eb6834", None, BOTH, ("shared",)),
        Level("l1", "L1 (vector L1 on AMD)", "#2a78d6", None, BOTH, ("l1",)),
        Level("scalarL1", "Scalar L1 (AMD)", "#2a78d6", "///", AMD, ("scalarL1",)),
        Level("constL1", "Constant L1 (NVIDIA)", "#4a3aa7", None, NVIDIA, ("constant", "l1")),
        Level("constL15", "Constant L1.5 (NVIDIA)", "#4a3aa7", "///", NVIDIA, ("constant", "l1.5")),
        Level("readOnly", "Read-only cache (NVIDIA)", "#1baf7a", None, NVIDIA, ("readOnly",)),
        Level("texture", "Texture cache (NVIDIA)", "#e87ba4", None, NVIDIA, ("texture",)),
        Level("l2", "L2", "#eda100", None, BOTH, ("l2",)),
        Level("l2seg", "L2 segment (NVIDIA)", "#eda100", "///", NVIDIA, ("l2",)),
        Level("l3", "L3 / Infinity Cache (AMD)", "#008300", None, AMD, ("l3",)),
        Level("main", "Main memory", "#e34948", None, BOTH, ("main",)),
    ]
}


# --------------------------------------------------------------------------- #
# GPU records                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Gpu:
    name: str          # display name (file stem, underscores -> spaces)
    vendor: str
    cc: tuple
    memory: dict
    compute: dict
    source: Path

    @property
    def short(self) -> str:
        for prefix in ("NVIDIA ", "AMD Instinct "):
            if self.name.startswith(prefix):
                return self.name[len(prefix):]
        return self.name

    def sort_key(self) -> tuple:
        return (self.vendor, self.cc, self.name)


def load_gpu_json(path: Path) -> Gpu:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read MT4G result '{path}': {exc}")
    for key in ("general", "memory"):
        if key not in data:
            raise SystemExit(f"error: '{path}' has no '{key}' section; not an MT4G result file")
    general = data["general"]
    vendor = general.get("vendor")
    if vendor not in ("AMD", "NVIDIA"):
        raise SystemExit(f"error: '{path}': unknown general.vendor {vendor!r}")
    cc = general.get("computeCapability", {})
    # The file stem is used as the name: some runtimes report a generic device
    # name (MI300A reports "AMD Radeon Graphics").
    return Gpu(
        name=path.stem.replace("_", " "),
        vendor=vendor,
        cc=(cc.get("major", 0), cc.get("minor", 0)),
        memory=data["memory"],
        compute=data.get("compute", {}),
        source=path,
    )


def node_of(gpu: Gpu, level: Level) -> dict:
    node = gpu.memory
    for part in level.json_node:
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}


def applicable(gpu: Gpu, level: Level) -> bool:
    """Whether MT4G has a benchmark for this level on this GPU at all.

    Vendor-specific levels follow the ``amd_``/``nvidia_`` split in
    ``src/benchmarks/``. L3 only exists where the runtime reports one (MI300+).
    """
    if gpu.vendor not in level.vendors:
        return False
    if level.key == "l3":
        return "l3" in gpu.memory
    return True


# --------------------------------------------------------------------------- #
# Metric extraction                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Value:
    status: str                    # ok | lowconf | missing | n/a
    value: Optional[float] = None
    source: str = "measured"       # measured | reported
    confidence: Optional[float] = None
    note: str = ""


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def get_latency(gpu: Gpu, level: Level) -> Value:
    if not applicable(gpu, level):
        return Value("n/a")
    lat = node_of(gpu, level).get("latency")
    if not isinstance(lat, dict) or not _finite_positive(lat.get("mean")):
        return Value("missing")
    if lat.get("unit", "cycles") != "cycles":
        raise SystemExit(f"error: {gpu.source}: {level.key} latency unit {lat.get('unit')!r} is not cycles")
    return Value("ok", float(lat["mean"]))


def get_cache_prop(gpu: Gpu, level: Level, metric: str) -> Value:
    """size / lineSize / fetchGranularity of one level, in bytes.

    MT4G stores p-chase measurements as ``{"size": .., "confidence": ..}`` and
    values queried from the runtime/driver as ``{"value": ..}``. Both are kept,
    but tagged, so figures can tell them apart.
    """
    if not applicable(gpu, level):
        return Value("n/a")
    if level.key == "l2seg":
        if metric != "size":
            return Value("n/a")
        metric = "segmentSize"
    entry = node_of(gpu, level).get(metric)
    if not isinstance(entry, dict):
        return Value("missing")
    if entry.get("unit", "bytes") != "bytes":
        raise SystemExit(f"error: {gpu.source}: {level.key}.{metric} unit {entry.get('unit')!r} is not bytes")
    if "size" in entry:
        conf = entry.get("confidence")
        if not _finite_positive(entry["size"]):
            return Value("missing")
        if isinstance(conf, (int, float)) and conf <= VALIDITY_THRESHOLD:
            return Value("lowconf", float(entry["size"]), "measured", conf)
        return Value("ok", float(entry["size"]), "measured", conf)
    if "value" in entry and _finite_positive(entry["value"]):
        return Value("ok", float(entry["value"]), "reported")
    return Value("missing")


# --------------------------------------------------------------------------- #
# Shared axis helpers                                                          #
# --------------------------------------------------------------------------- #
def style_gpu_axis(ax, gpus: list[Gpu], show_labels: bool = True, vendor_labels: bool = True) -> None:
    ax.set_xticks(range(len(gpus)))
    ax.set_xticklabels([g.short for g in gpus] if show_labels else [], rotation=28, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(gpus) - 0.4)
    ax.tick_params(axis="x", length=0)
    # Vendor separators and names.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    vendors = [g.vendor for g in gpus]
    start = 0
    for i in range(1, len(gpus) + 1):
        if i == len(gpus) or vendors[i] != vendors[start]:
            if vendor_labels:
                ax.text((start + i - 1) / 2, 1.01, vendors[start], transform=trans, ha="center",
                        va="bottom", fontsize=13, fontweight="bold", color=TEXT_MUTED)
            if i < len(gpus):
                ax.axvline(i - 0.5, color="0.6", linewidth=0.8, linestyle="-")
            start = i


def fmt_bytes(b: float) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= scale:
            v = b / scale
            return f"{v:.0f} {unit}" if abs(v - round(v)) < 0.05 else f"{v:.1f} {unit}"
    return f"{b:.0f} B"


def fmt_gib(v: float) -> str:
    return f"{v:.0f}" if v >= 100 else f"{v:.3g}"


def save(fig, outdir: Path, name: str, written: list[Path]) -> None:
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    written.append(path)
    print(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")


def level_handles(levels: list[Level], with_reported: bool = False, with_missing: bool = True) -> list:
    hs = [Patch(facecolor=lv.color, edgecolor="white", hatch=lv.hatch, label=lv.label) for lv in levels]
    if with_reported:
        hs.append(Patch(facecolor="white", edgecolor="0.35", linewidth=1.2,
                        label="Hollow = reported by runtime/driver, not measured"))
    if with_missing:
        hs.append(Line2D([], [], linestyle="none", marker="$✕$", color=TEXT_MUTED, markersize=8,
                         label="✕ = applicable but no valid measurement"))
    return hs


# --------------------------------------------------------------------------- #
# Grouped bars: GPU on x, one bar per applicable level                         #
# --------------------------------------------------------------------------- #
def grouped_bars(ax, gpus: list[Gpu], levels: list[Level], getter: Callable[[Gpu, Level], Value],
                 value_fmt: Callable[[float], str] = lambda v: f"{v:.0f}", annotate: bool = True) -> set:
    """Draw one bar per level that applies to each GPU's vendor.

    Missing / low-confidence values get a marker at the baseline instead of a
    zero-height bar. Returns the levels that were actually drawn.
    """
    slots = {g.name: [lv for lv in levels if getter(g, lv).status != "n/a"] for g in gpus}
    max_slots = max((len(s) for s in slots.values()), default=1) or 1
    width = 0.84 / max_slots
    drawn = set()
    for i, g in enumerate(gpus):
        s = slots[g.name]
        x0 = i - width * len(s) / 2 + width / 2
        for j, lv in enumerate(s):
            x = x0 + j * width
            v = getter(g, lv)
            if v.status == "ok":
                reported = v.source == "reported"
                ax.bar(x, v.value, width=width * 0.92, color="white" if reported else lv.color,
                       edgecolor=lv.color if reported else "white", linewidth=1.3 if reported else 0.4,
                       hatch=lv.hatch, zorder=2)
                if annotate:
                    ax.annotate(value_fmt(v.value), (x, v.value), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.5, rotation=90, color="#3b3a37")
                drawn.add(lv.key)
            else:
                ax.plot([x], [0], marker="$✕$", color=TEXT_MUTED, markersize=7, zorder=3,
                        clip_on=False)
                drawn.add("missing")
    return drawn


# --------------------------------------------------------------------------- #
# Latency                                                                      #
# --------------------------------------------------------------------------- #
LAT_PANELS = [
    ("Level-1 caches, constant caches and shared memory (close to the SM/CU)",
     ["shared", "l1", "scalarL1", "constL1", "constL15", "readOnly", "texture"]),
    ("L2, L3 and main memory", ["l2", "l3", "main"]),
]


def plot_latency(gpus: list[Gpu], outdir: Path, written: list[Path]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12.5))
    for ax, (title, keys) in zip(axes, LAT_PANELS):
        levels = [LEVELS[k] for k in keys]
        drawn = grouped_bars(ax, gpus, levels, get_latency)
        ymax = max((get_latency(g, lv).value or 0) for g in gpus for lv in levels)
        ax.set_ylim(0, ymax * 1.16)
        ax.set_ylabel("Mean latency (cycles)")
        ax.set_title(title, loc="left", pad=16)
        style_gpu_axis(ax, gpus)
        ax.legend(handles=level_handles([lv for lv in levels if lv.key in drawn],
                                        with_missing="missing" in drawn),
                  loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    for ax in axes:
        ax.set_xlabel("GPU model")
    fig.suptitle("Memory access latency by GPU and memory hierarchy level "
                 "(mean, linear y-axis)", x=0.01, ha="left", fontsize=16.5,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "latency_mean_by_gpu", written)


# --------------------------------------------------------------------------- #
# Cache properties                                                             #
# --------------------------------------------------------------------------- #
SIZE_PANELS = [
    ("Per-SM/CU caches (measured)", ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture"]),
    ("Device-level caches", ["l2", "l2seg", "l3"]),
]
LINE_LEVELS = ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture", "l2", "l3"]
FETCH_LEVELS = ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture", "l2"]


def plot_cache_size(gpus: list[Gpu], outdir: Path, written: list[Path]) -> None:
    """Dot plot on a log2 axis: sizes span 2 KiB .. 256 MiB, so bars on a
    linear axis would hide every per-SM cache; bars on a log axis have no
    meaningful baseline."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12.5))
    for ax, (title, keys) in zip(axes, SIZE_PANELS):
        levels = [LEVELS[k] for k in keys]
        getter = lambda g, lv: get_cache_prop(g, lv, "size")
        slots = {g.name: [lv for lv in levels if getter(g, lv).status != "n/a"] for g in gpus}
        width = 0.84 / max(len(s) for s in slots.values())
        vals, drawn, any_reported, missing_x = [], set(), False, []
        for i, g in enumerate(gpus):
            s = slots[g.name]
            x0 = i - width * len(s) / 2 + width / 2
            for j, lv in enumerate(s):
                x, v = x0 + j * width, getter(g, lv)
                if v.status == "ok":
                    reported = v.source == "reported"
                    any_reported |= reported
                    ax.vlines(x, 1, v.value, color=lv.color, linewidth=1.0, alpha=0.35, zorder=1)
                    ax.scatter([x], [v.value], s=80, zorder=3, marker="s" if lv.hatch else "o",
                               facecolor="white" if reported else lv.color, edgecolor=lv.color,
                               linewidth=1.6 if reported else 0.6)
                    vals.append(v.value)
                    drawn.add(lv.key)
                else:  # missing or low confidence: an x near the axis floor, drawn below
                    missing_x.append(x)
        ax.set_yscale("log", base=2)
        lo = 2 ** math.floor(math.log2(min(vals)) - 0.5)
        hi = 2 ** math.ceil(math.log2(max(vals)) + 0.7)
        ax.set_ylim(lo, hi)
        ticks = [2 ** e for e in range(int(math.log2(lo)), int(math.log2(hi)) + 1)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([fmt_bytes(t) for t in ticks])
        ax.minorticks_off()
        for x in missing_x:
            ax.plot([x], [lo * 1.12], marker="$✕$", color=TEXT_MUTED, markersize=7, zorder=3)
        ax.set_ylabel("Cache size (bytes, log₂ scale)")
        ax.set_title(title, loc="left", pad=16)
        style_gpu_axis(ax, gpus)
        hs = [Line2D([], [], linestyle="none", marker="s" if lv.hatch else "o", markersize=10,
                     markerfacecolor=lv.color, markeredgecolor=lv.color, label=lv.label)
              for lv in levels if lv.key in drawn]
        if any_reported:
            hs.append(Line2D([], [], linestyle="none", marker="o", markersize=10, markerfacecolor="white",
                             markeredgecolor="0.35", label="Hollow = reported by runtime/driver"))
        if missing_x:
            hs.append(Line2D([], [], linestyle="none", marker="$✕$", color=TEXT_MUTED, markersize=8,
                             label="✕ = no valid measurement\n(missing or confidence ≤ 0.5)"))
        ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    for ax in axes:
        ax.set_xlabel("GPU model")
    fig.suptitle("Cache size by GPU and memory hierarchy level (log₂ y-axis)", x=0.01, ha="left",
                 fontsize=16.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "cache_size_by_gpu", written)


def plot_bytes_bars(gpus: list[Gpu], outdir: Path, written: list[Path], metric: str, keys: list[str],
                    name: str, title: str, ylabel: str) -> None:
    levels = [LEVELS[k] for k in keys]
    getter = lambda g, lv: get_cache_prop(g, lv, metric)
    fig, ax = plt.subplots(figsize=(16, 6.6))
    drawn = grouped_bars(ax, gpus, levels, getter)
    vals = [getter(g, lv).value for g in gpus for lv in levels if getter(g, lv).status == "ok"]
    ymax = max(vals)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_yticks(range(0, int(ymax) + 1, 32))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("GPU model")
    style_gpu_axis(ax, gpus)
    any_reported = any(getter(g, lv).source == "reported" and getter(g, lv).status == "ok"
                       for g in gpus for lv in levels)
    ax.legend(handles=level_handles([lv for lv in levels if lv.key in drawn], with_reported=any_reported,
                                    with_missing="missing" in drawn),
              loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.set_title(title, loc="left", pad=18, fontsize=16.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, name, written)


# --------------------------------------------------------------------------- #
# Bandwidth data: configuration model and parser                               #
# --------------------------------------------------------------------------- #
# What each benchmark in src/benchmarks/bandwidth/ sweeps, transcribed from the
# sources. With a GPU's "compute" section it gives the exact launch configurations
# the benchmark uses there, so figures exist before any measurement does. The
# parser reads the grids written by util::writeBandwidthGridToCSV
# (<device>__<benchmark>_<Read|Write>_[<N>KiB_<dyn|stat>_]BW_Grid.csv, GiB/s);
# the bandwidth fields inside the MT4G JSON files are never read.
BENCH_SRC = REPO_ROOT / "src" / "benchmarks" / "bandwidth"

# main.cpp: MIN_EXPECTED_LINE_SIZE (and VALIDITY_THRESHOLD above), used to size the
# constant-cache working sets (see _constant_l1_thread_cap / _constant_l15_thread_cap).
MIN_EXPECTED_LINE_SIZE = 4


# --------------------------------------------------------------------------- #
# Benchmark specification                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bench:
    key: str               # unique id, used in file names
    level: str             # hierarchy level key (colour), see plot_results.LEVELS
    label: str             # human-readable hierarchy name
    direction: str         # read | write
    vendors: frozenset
    device_wide: bool      # True: grid sweeps blocks = numSM * 2^k; False: one block (CU pinned)
    token: str             # benchmark token in the grid file name (main.cpp)
    alloc: Optional[str]   # LDS allocation suffix (dyn | stat) or None
    sources: tuple         # files in src/benchmarks/bandwidth/ implementing it
    single_rep: bool = False                                   # main memory: one streaming pass
    thread_cap: Optional[Callable[[dict], Optional[int]]] = None  # data-dependent max threads
    requires: Optional[Callable[[dict], bool]] = None           # level exists on this GPU


def _conf_ok(entry) -> bool:
    return isinstance(entry, dict) and (entry.get("confidence") or 0) > VALIDITY_THRESHOLD


def _constant_l1_thread_cap(memory: dict) -> Optional[int]:
    """nvidia_constantL1ReadBandwidth.cpp capNumThreads: min(maxThreads, bytes / 4),
    with bytes = constantL1Size / 2 (main.cpp). Unknown when the size is not valid
    (the benchmark is then skipped)."""
    size = memory.get("constant", {}).get("l1", {}).get("size")
    if not _conf_ok(size):
        return None
    return (size["size"] // 2) // 4


def _constant_l15_thread_cap(memory: dict) -> Optional[int]:
    """nvidia_constantL15ReadBandwidth.cpp capNumThreads: min(maxThreads, bytes / stride),
    bytes = L1.5 size / 2 if valid else 32 KiB, stride = L1.5 fetch granularity if
    valid else MIN_EXPECTED_LINE_SIZE (main.cpp)."""
    l15 = memory.get("constant", {}).get("l1.5", {})
    size, fg = l15.get("size"), l15.get("fetchGranularity")
    nbytes = size["size"] // 2 if _conf_ok(size) else 32 * 1024
    stride = fg["size"] if _conf_ok(fg) else MIN_EXPECTED_LINE_SIZE
    return nbytes // stride


def _has_l3(memory: dict) -> bool:
    return "l3" in memory


def _rw(key, level, label, vendors, device_wide, token, src_r, src_w, alloc=None, **kw):
    out = [Bench(f"{key}", level, label, "read", vendors, device_wide, token, alloc, src_r, **kw)]
    if src_w:
        out.append(Bench(f"{key}", level, label, "write", vendors, device_wide, token, alloc, src_w, **kw))
    return out


# Order = README order (memory side first, then the per-SM/CU levels).
BENCHMARKS: list[Bench] = [
    *_rw("main", "main", "Main memory", BOTH, True, "MainMemory",
         ("mainMemoryReadBandwidth.cpp",), ("mainMemoryWriteBandwidth.cpp",), single_rep=True),
    *_rw("l3", "l3", "L3 / Infinity Cache", AMD, True, "L3",
         ("amd_l3ReadBandwidth.cpp",), ("amd_l3WriteBandwidth.cpp",), requires=_has_l3),
    *_rw("l2", "l2", "L2 cache", BOTH, True, "L2",
         ("l2ReadBandwidth.cpp",), ("l2WriteBandwidth.cpp",)),
    *_rw("l1", "l1", "L1 cache", BOTH, False, "L1",
         ("l1ReadBandwidth.cpp",), ("l1WriteBandwidth.cpp",)),
    *_rw("scalarL1", "scalarL1", "Scalar L1 cache", AMD, False, "sL1d",
         ("amd_sL1ReadBandwidth.cpp",), ("amd_sL1WriteBandwidth.cpp",)),
    *_rw("constL15", "constL15", "Constant L1.5 cache", NVIDIA, False, "ConstantL1.5",
         ("nvidia_constantL15ReadBandwidth.cpp",), None, thread_cap=_constant_l15_thread_cap),
    *_rw("constL1", "constL1", "Constant L1 cache", NVIDIA, False, "ConstantL1",
         ("nvidia_constantL1ReadBandwidth.cpp",), None, thread_cap=_constant_l1_thread_cap),
    *_rw("readOnly", "readOnly", "Read-only cache", NVIDIA, False, "ReadOnly",
         ("nvidia_readOnlyReadBandwidth.cpp",), None),
    *_rw("texture", "texture", "Texture cache", NVIDIA, False, "Texture",
         ("nvidia_textureReadBandwidth.cpp",), None),
    *_rw("sharedDyn", "shared", "Shared memory / LDS (dynamic allocation)", BOTH, False, "LDS",
         ("sharedReadBandwidth.cpp",), ("sharedWriteBandwidth.cpp",), alloc="dyn"),
    *_rw("sharedStat", "shared", "Shared memory / LDS (static allocation)", BOTH, False, "LDS",
         ("sharedReadBandwidthStatic.cpp",), ("sharedWriteBandwidthStatic.cpp",), alloc="stat"),
]
# The L1 grid token is "vL1d" on AMD and "L1" on NVIDIA (main.cpp, l1Label).
TOKEN_ALIASES = {"vL1d": "L1"}


def check_sources() -> list[str]:
    """Benchmarks whose source files are missing: the spec above is stale."""
    return [f"{b.key}/{b.direction}: {s}" for b in BENCHMARKS for s in b.sources if not (BENCH_SRC / s).is_file()]


def read_reps() -> Optional[tuple[int, int]]:
    """MIN_REPS / MAX_REPS from src/config.hpp (repetitions per launch)."""
    try:
        text = (REPO_ROOT / "src" / "config.hpp").read_text()
    except OSError:
        return None
    vals = {m[1]: int(m[2]) for m in re.finditer(r"(MIN_REPS|MAX_REPS)\s*=\s*(\d+)", text)}
    return (vals["MIN_REPS"], vals["MAX_REPS"]) if len(vals) == 2 else None


# --------------------------------------------------------------------------- #
# Configurations                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, order=True)
class Config:
    """One launch configuration. ``sm_multiple`` is the block count divided by
    the SM/CU count for device-wide sweeps, and None for single-block benchmarks."""
    sm_multiple: Optional[int]
    threads: int

    def slug(self) -> str:
        blocks = f"{self.sm_multiple}xSM" if self.sm_multiple else "1"
        return f"blocks-{blocks}_threads-{self.threads}"

    def text(self) -> str:
        blocks = f"{self.sm_multiple} × #SM/CU blocks" if self.sm_multiple else "1 block"
        return f"{blocks}, {self.threads} threads/block"


def _pow2_range(lo: int, hi: int) -> list[int]:
    out, v = [], lo
    while v <= hi:
        out.append(v)
        v *= 2
    return out


def expected_configs(bench: Bench, vendor: str, compute: dict, memory: dict) -> Optional[set[Config]]:
    """Configurations ``bench`` launches on this GPU, or None if it does not run there.

    Thread sweeps go warpSize .. maxThreadsPerBlock doubling (every benchmark);
    device-wide block sweeps go numSM .. numSM * maxBlocksPerMultiProcessor
    doubling (l2/l3/mainMemory *Bandwidth.cpp)."""
    if vendor not in bench.vendors or (bench.requires and not bench.requires(memory)):
        return None
    warp, tmax = compute.get("warpSize"), compute.get("maxThreadsPerBlock")
    if not warp or not tmax:
        return None
    if bench.thread_cap:
        cap = bench.thread_cap(memory)
        if cap is not None:
            tmax = min(tmax, cap)
    threads = _pow2_range(warp, tmax)
    if not bench.device_wide:
        return {Config(None, t) for t in threads}
    bmax = compute.get("maxBlocksPerMultiProcessor")
    if not bmax:
        return None
    return {Config(k, t) for k in _pow2_range(1, bmax) for t in threads}


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
GRID_RE = re.compile(
    r"^(?P<base>.+)__(?P<bench>[A-Za-z0-9.]+)_(?P<dir>Read|Write)_"
    r"(?:(?P<kib>\d+)KiB_(?P<alloc>dyn|stat)_)?BW_Grid\.csv$"
)


@dataclass
class Measurement:
    bench: Bench
    config: Config
    blocks: int
    reps: int
    gib_s: float
    source: Path


@dataclass
class Run:
    device: str            # device name as written by MT4G (JSON general.name)
    vendor: str
    compute: dict
    json_path: Path
    meta: dict
    memory: dict = field(default_factory=dict)
    measurements: list[Measurement] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def norm_name(name: str) -> str:
    return re.sub(r"[\s_]+", "_", name.strip()).lower()


def _read_grid(path: Path) -> list[tuple[int, int, int, float]]:
    """(blocks, threads, reps, GiB/s) rows; the wide single-block format gets blocks=1."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")
    header = [h.strip().lower() for h in lines[0].split(",")]
    rows = []
    if header == ["blocks", "threads", "reps", "bandwidth"]:
        for ln in lines[1:]:
            b, t, r, v = ln.split(",")
            rows.append((int(b), int(t), int(r), float(v)))
    elif header[0] == "threads" and len(header) > 1 and all(h.isdigit() for h in header[1:]):
        reps = [int(h) for h in header[1:]]
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) != len(header):
                raise ValueError(f"row has {len(parts)} fields, header has {len(header)}")
            rows.extend((1, int(parts[0]), r, float(v)) for r, v in zip(reps, parts[1:]))
    else:
        raise ValueError(f"unrecognised header {lines[0]!r}")
    return rows


def _find_run_json(grid_dir: Path, device: str, root: Path) -> Optional[Path]:
    """MT4G writes <location>/<device>.json and the grids to <location>/results/<device>/;
    look in the grid directory and upwards (up to root) for the matching JSON."""
    d = grid_dir
    while True:
        for j in sorted(d.glob("*.json")):
            if norm_name(j.stem) == norm_name(device):
                return j
        if d == root or root not in d.parents:
            return None
        d = d.parent


def load_runs(root: Path, exclude: Optional[Path] = None) -> tuple[list[Run], list[str]]:
    """Find and validate every bandwidth grid below ``root``.

    Returns the runs and a list of warnings for anything that was rejected
    (unknown names, missing JSON, bad rows, configurations the benchmark
    cannot produce). Nothing is rejected silently."""
    warnings: list[str] = []
    reps = read_reps()
    grouped: dict[tuple, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*_BW_Grid.csv")):
        if exclude and exclude in p.parents:
            continue
        m = GRID_RE.match(p.name)
        if not m:
            warnings.append(f"{p}: file name does not follow the bandwidth grid scheme; ignored")
            continue
        grouped[(p.parent, m["base"])].append(p)

    by_token = {(b.token, b.alloc, b.direction): b for b in BENCHMARKS}
    runs = []
    for (d, device), files in sorted(grouped.items()):
        jpath = _find_run_json(d, device, root)
        if jpath is None:
            warnings.append(f"{d}: no MT4G JSON '{device}.json' next to or above the grids "
                            "(needed for vendor and SM/CU count); grids ignored")
            continue
        try:
            raw = json.loads(jpath.read_text())
            general, compute = raw["general"], raw["compute"]
            vendor = general["vendor"]
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f"{jpath}: not a usable MT4G result ({exc}); grids ignored")
            continue
        run = Run(general.get("name", device), vendor, compute, jpath, raw.get("meta", {}),
                  memory=raw.get("memory", {}), files=files)
        sms = compute.get("multiProcessorCount")
        seen: set = set()
        for f in files:
            m = GRID_RE.match(f.name)
            token = TOKEN_ALIASES.get(m["bench"], m["bench"])
            bench = by_token.get((token, m["alloc"], m["dir"].lower()))
            if bench is None:
                warnings.append(f"{f}: unknown benchmark '{m['bench']}'; ignored")
                continue
            if vendor not in bench.vendors:
                warnings.append(f"{f}: {bench.label} is not a {vendor} benchmark; ignored")
                continue
            try:
                rows = _read_grid(f)
            except ValueError as exc:
                warnings.append(f"{f}: {exc}; ignored")
                continue
            expected = expected_configs(bench, vendor, compute, raw.get("memory", {})) or set()
            for blocks, threads, r, v in rows:
                if not (math.isfinite(v) and v > 0):
                    continue  # the wide writer emits 0.0 for cells it never measured
                if bench.device_wide:
                    if not sms or blocks % sms:
                        warnings.append(f"{f}: {blocks} blocks is not a multiple of {sms} SMs/CUs; row ignored")
                        continue
                    cfg = Config(blocks // sms, threads)
                else:
                    if blocks != 1:
                        warnings.append(f"{f}: single-block benchmark with {blocks} blocks; row ignored")
                        continue
                    cfg = Config(None, threads)
                want_reps = (1, 1) if bench.single_rep else reps
                if want_reps and not want_reps[0] <= r <= want_reps[1]:
                    warnings.append(f"{f}: reps={r} outside MIN_REPS..MAX_REPS {want_reps}; row ignored")
                    continue
                if cfg not in expected:
                    warnings.append(f"{f}: {cfg.text()} is not a configuration this benchmark launches "
                                    f"on this GPU; kept, but check the benchmark spec")
                key = (bench.key, bench.direction, cfg, r)
                if key in seen:
                    warnings.append(f"{f}: duplicate {cfg.text()} reps={r}; row ignored")
                    continue
                seen.add(key)
                run.measurements.append(Measurement(bench, cfg, blocks, r, v, f))
        runs.append(run)
    return runs, warnings


# --------------------------------------------------------------------------- #
# Bandwidth plots: one figure per direction and launch configuration           #
# --------------------------------------------------------------------------- #
BW_DPI = 100
# Two families, like the two latency panels. Single-block levels only sweep threads,
# device-wide levels sweep blocks x threads, so the two never share a configuration.
BW_FAMILIES = {False: ("per-sm", "per-SM/CU levels (single block)"),
               True: ("device", "device-wide levels (main memory, L2, L3)")}
BENCH_HATCH = {"scalarL1": "///", "constL15": "///", "sharedStat": "..."}


@dataclass
class BwColumn:
    """One GPU on the bandwidth x-axis, with the runs measured on it."""
    name: str
    vendor: str
    compute: dict
    memory: dict
    sort_key: tuple
    runs: list = field(default_factory=list)

    @property
    def short(self) -> str:
        return Gpu.short.fget(self)


def bw_columns(gpus: list[Gpu], runs: list[Run]) -> list[BwColumn]:
    """Every sample GPU, plus any GPU that only has bandwidth runs. A run is matched to a
    sample GPU by device name; its own JSON then defines the configurations."""
    cols = {norm_name(g.name): BwColumn(g.name, g.vendor, g.compute, g.memory, g.sort_key()) for g in gpus}
    for r in runs:
        key = next((k for k in (norm_name(r.device), norm_name(r.json_path.stem)) if k in cols), None)
        if key is None:
            key = norm_name(r.device)
            cols[key] = BwColumn(r.device, r.vendor, r.compute, r.memory, (r.vendor, (0, 0), r.device))
        col = cols[key]
        if not col.runs:
            col.compute, col.memory = r.compute, r.memory
        col.runs.append(r)
    return sorted(cols.values(), key=lambda c: c.sort_key)


@dataclass
class BwPlot:
    direction: str
    device_wide: bool
    config: Config
    path: Path
    benches: list      # levels in this figure (legend order)
    columns: list      # GPUs on the x-axis
    slots: list        # per column: [(bench, status, [Measurement])], status ok | missing | n/a

    @property
    def n_points(self) -> int:
        return sum(len(ms) for col in self.slots for _, _, ms in col)


def bw_plan(columns: list[BwColumn], outdir: Path) -> list[BwPlot]:
    """One figure per direction, family and launch configuration used by any GPU.

    A level gets a slot for a GPU if MT4G runs it there (vendor, L3 present).
    The slot is n/a if this configuration is not launched on that GPU and
    'missing' (left empty) if it is launched but there is no measurement."""
    plots = []
    for direction in ("read", "write"):
        for device_wide in (False, True):
            benches = sorted((b for b in BENCHMARKS if b.direction == direction and b.device_wide == device_wide),
                             key=lambda b: list(LEVELS).index(b.level))  # same order as the latency figure
            cols = [c for c in columns if any(c.vendor in b.vendors for b in benches)]
            # per column: [(bench, expected configs, measurements)]
            info = [[(b, expected_configs(b, c.vendor, c.compute, c.memory),
                      [m for r in c.runs for m in r.measurements if m.bench is b])
                     for b in benches if c.vendor in b.vendors and (not b.requires or b.requires(c.memory))]
                    for c in cols]
            configs = {cfg for col in info for _, exp, ms in col for cfg in (exp or set()) | {m.config for m in ms}}
            used = [b for b in benches if any(b is bb for col in info for bb, _, _ in col)]
            slug, _ = BW_FAMILIES[device_wide]
            for cfg in sorted(configs, key=lambda c: (c.sm_multiple or 0, c.threads)):
                slots = []
                for col in info:
                    row = []
                    for b, exp, ms in col:
                        hit = [m for m in ms if m.config == cfg]
                        status = "ok" if hit else ("n/a" if exp is None or cfg not in exp else "missing")
                        row.append((b, status, hit))
                    slots.append(row)
                name = f"bw_{direction}_{slug}_{cfg.slug()}"
                plots.append(BwPlot(direction, device_wide, cfg, outdir / direction / f"{name}.png",
                                    used, cols, slots))
    return plots


def plot_bw(plan: BwPlot) -> int:
    cfg = plan.config
    fig, ax = plt.subplots(figsize=(16, 7))
    max_slots = max(len(col) for col in plan.slots)
    width = 0.84 / max_slots
    vals = [m.gib_s for col in plan.slots for _, _, ms in col for m in ms]
    plotted = 0
    for i, col in enumerate(plan.slots):
        x0 = i - width * len(col) / 2 + width / 2
        for j, (b, status, ms) in enumerate(col):
            x = x0 + j * width
            if status == "n/a":
                ax.text(x, 0, " n/a", rotation=90, ha="center", va="bottom", fontsize=9, color=TEXT_MUTED)
            sub = width * 0.92 / max(len(ms), 1)
            for k, m in enumerate(ms):  # several runs of one GPU: side by side, never merged
                xm = x - width * 0.46 + sub * (k + 0.5)
                ax.bar(xm, m.gib_s, width=sub, color=LEVELS[b.level].color, edgecolor="white", linewidth=0.4,
                       hatch=BENCH_HATCH.get(b.key), zorder=2)
                ax.annotate(fmt_gib(m.gib_s), (xm, m.gib_s), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, rotation=90, color="#3b3a37")
                plotted += 1
    if vals:
        ax.set_ylim(0, max(vals) * 1.22)
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(0.5, 0.55, "TEMPLATE — no measurements yet", transform=ax.transAxes, ha="center",
                va="center", fontsize=22, color="0.75", fontweight="bold")
    style_gpu_axis(ax, plan.columns)
    if plan.device_wide:  # the absolute block count this configuration means on each GPU
        labels = []
        for c, col in zip(plan.columns, plan.slots):
            sms = c.compute.get("multiProcessorCount")
            launched = any(status != "n/a" for _, status, _ in col)
            labels.append(f"{c.short} · {cfg.sm_multiple * sms} blocks" if sms and launched else c.short)
        ax.set_xticklabels(labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlabel("GPU model")
    ax.set_ylabel("Bandwidth (GiB/s)")
    _, family = BW_FAMILIES[plan.device_wide]
    ax.set_title(f"{plan.direction.capitalize()} bandwidth — {family}\n{cfg.text()}",
                 loc="left", pad=26, fontsize=16.5, fontweight="bold")
    hs = [Patch(facecolor=LEVELS[b.level].color, edgecolor="white", hatch=BENCH_HATCH.get(b.key), label=b.label)
          for b in plan.benches]
    hs.append(Line2D([], [], linestyle="none", label="n/a = configuration not launched on this GPU"))
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    fig.tight_layout()
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=BW_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    # Many flat-colour charts: an adaptive 256-colour palette cuts the size ~4x with no visible change.
    Image.open(buf).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
        plan.path, optimize=True)
    return plotted


def plot_bandwidth(columns: list[BwColumn], outdir: Path) -> tuple[list[BwPlot], int]:
    bw_dir = outdir / "bandwidth"
    for old in bw_dir.rglob("bw_*.png"):  # drop figures of configurations that no longer exist
        old.unlink()
    plans = bw_plan(columns, bw_dir)
    plotted = sum(plot_bw(p) for p in plans)
    print(f"  wrote {len(plans)} bandwidth plots to {bw_dir}")
    return plans, plotted


# --------------------------------------------------------------------------- #
# README                                                                       #
# --------------------------------------------------------------------------- #
def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def cell(v: Value, fmt: Callable[[float], str]) -> str:
    if v.status == "n/a":
        return ""
    if v.status == "missing":
        return "—"
    if v.status == "lowconf":
        return f"~~{v.value:.0f} B~~ (conf. {v.confidence:.2f})"
    return fmt(v.value) + ("†" if v.source == "reported" else "")


def metric_table(gpus: list[Gpu], keys: list[str], getter, fmt) -> str:
    levels = [LEVELS[k] for k in keys]
    levels = [lv for lv in levels if any(getter(g, lv).status != "n/a" for g in gpus)]
    rows = [[g.name] + [cell(getter(g, lv), fmt) for lv in levels] for g in gpus]
    return md_table(["GPU"] + [lv.label for lv in levels], rows)


REPORTED_NOTE = ("Hollow markers and bars (and † in the tables) are values reported by the HIP runtime "
                 "rather than measured.")


def values_toggle(metric: str, gpus: list[Gpu], keys: list[str], getter, fmt) -> str:
    """Collapsible value table; the runtime-reported note is added only if the table has such values."""
    reported = any(getter(g, LEVELS[k]).status == "ok" and getter(g, LEVELS[k]).source == "reported"
                   for g in gpus for k in keys)
    note = f"{REPORTED_NOTE}\n\n" if reported else ""
    return (f"<details><summary>Values ({metric}; — = no valid value; blank = not applicable)</summary>\n\n"
            + note + metric_table(gpus, keys, getter, fmt) + "\n\n</details>\n")


def img(path: Path, readme: Path, alt: str) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, readme.parent)).as_posix()})"


def bw_readme(s: list, n: int, plans: list[BwPlot], runs: list[Run], readme: Path) -> None:
    rel = lambda p: Path(os.path.relpath(p, readme.parent)).as_posix()
    measured = [r for r in runs if r.measurements]
    if not measured:
        s.append("> **Template plots — no compatible measurements yet.** The figures below only show which "
                 "GPUs, memory levels and launch configurations the benchmarks cover; no performance "
                 "comparison can be made from them. Add new-format results (the `*_BW_Grid.csv` files from "
                 "`mt4g -g` plus the MT4G JSON of the same run) anywhere under `sample_results/` and rerun "
                 "the script.\n")
    else:
        def info(r):
            extra = ", ".join(str(r.meta[k]) for k in ("mt4gVersion", "timestamp") if r.meta.get(k))
            return f"**{r.device}**" + (f" ({extra})" if extra else "")
        s.append("> Measurements are available for: " + ", ".join(info(r) for r in measured)
                 + ". Plots without any bar are templates for configurations not measured yet.\n")
    s.append("- One plot per direction and launch configuration; bars are memory levels, one per "
             "measurement. Nothing is averaged.\n"
             "- **Per-SM/CU levels** (L1, scalar L1, constant, read-only, texture, shared memory/LDS) run a "
             "single block on one SM/CU (CU count pinned to 1), so block count is not swept: one plot per "
             "thread count.\n"
             "- **Device-wide levels** (main memory, L2, L3) sweep the block count as k × #SM/CU (k = 1, 2, "
             "4, … up to the GPU's max blocks per SM/CU): one plot per block count and thread count. The "
             "absolute block count is next to each GPU name.\n"
             "- *n/a*: the configuration is not launched on that GPU (e.g. k above its max blocks per SM/CU). "
             "An empty slot: not measured yet, or skipped by the early stop of the device-wide sweeps.\n")
    for d, sub in (("read", 1), ("write", 2)):
        s.append(f"### {n}.{sub} {d.capitalize()} bandwidth\n")
        for i, device_wide in enumerate((False, True), 1):
            ps = [p for p in plans if p.direction == d and p.device_wide == device_wide]
            if not ps:
                continue
            _, family = BW_FAMILIES[device_wide]
            s.append(f"#### {n}.{sub}.{i} {family[0].upper() + family[1:]}\n")
            s.append(f"*Levels: {', '.join(b.label for b in ps[0].benches)}. "
                     + ("Rows: block count (× #SM/CU); columns: threads per block. " if device_wide
                        else "Columns: threads per block. ") + "Click a plot to enlarge.*\n")
            threads = sorted({p.config.threads for p in ps})
            rows = sorted({p.config.sm_multiple for p in ps}, key=lambda k: k or 0)
            by_cfg = {(p.config.sm_multiple, p.config.threads): p for p in ps}
            html = ["<table>", "<tr><th>Blocks</th>" + "".join(f"<th>{t} threads</th>" for t in threads) + "</tr>"]
            for k in rows:
                cells = []
                for t in threads:
                    p = by_cfg.get((k, t))
                    if p is None:
                        cells.append("<td></td>")
                    else:
                        src = rel(p.path)
                        cells.append(f'<td><a href="{src}"><img src="{src}" width="150" '
                                     f'alt="{d} bandwidth, {family}, {p.config.text()}"></a></td>')
                html.append(f"<tr><th>{f'{k} × #SM/CU' if k else '1'}</th>" + "".join(cells) + "</tr>")
            html.append("</table>")
            s.append("\n".join(html) + "\n")


def write_readme(readme: Path, gpus: list[Gpu], runs: list[Run], figs: dict[str, Path],
                 plans: list[BwPlot]) -> None:
    rel = lambda p: Path(os.path.relpath(p, readme.parent)).as_posix()
    n_amd = sum(g.vendor == "AMD" for g in gpus)
    sections = ["Overview", "Memory latency", "Cache properties"] + (["Bandwidth"] if plans else [])
    num = {name: i + 1 for i, name in enumerate(sections)}
    head = lambda name: f"## {num[name]}. {name}\n"
    anchor = lambda name: f"#{num[name]}-{name.lower().replace(' ', '-')}"

    s = []
    s.append("# MT4G sample results\n")
    s.append("This page is generated by `scripts/plot_results.py`; edit the script, not this file. "
             "Re-run it after adding results (needs Python 3.9+ with matplotlib):\n\n```bash\n"
             "python3 scripts/plot_results.py\n```\n")
    s.append("## Contents\n\n" + "\n".join(f"{num[n]}. [{n}]({anchor(n)})" for n in sections) + "\n")

    s.append(head("Overview"))
    s.append(f"One MT4G result file per GPU: {len(gpus)} GPUs, {len(gpus) - n_amd} NVIDIA and {n_amd} AMD.\n")
    s.append(md_table(["GPU", "Vendor", "Compute capability (HIP)", "SMs/CUs", "Result file"],
                      [[g.name, g.vendor, f"{g.cc[0]}.{g.cc[1]}", str(g.compute.get("multiProcessorCount", "?")),
                        f"[`{g.source.name}`]({rel(g.source).replace(' ', '%20')})"] for g in gpus]) + "\n")
    metrics = ["Memory latency: mean, in cycles", "Cache size, in bytes", "Cache line size, in bytes",
               "Fetch granularity, in bytes"]
    if plans:
        metrics.append("Read and write bandwidth, in GiB/s, for every swept launch configuration"
                       + ("" if any(r.measurements for r in runs) else " (template plots, no measurements yet)"))
    s.append("Metrics covered:\n\n" + "\n".join(f"- {m}" for m in metrics) + "\n")

    s.append(head("Memory latency"))
    s.append(img(figs["latency"], readme, "Mean memory latency per GPU and hierarchy level") + "\n")
    s.append(values_toggle("mean latency, cycles", gpus, [k for k in LEVELS if k != "l2seg"], get_latency,
                           lambda v: f"{v:.1f}"))

    s.append(head("Cache properties"))
    n = num["Cache properties"]
    sizes = [v.value for g in gpus for lv in LEVELS.values()
             for v in [get_cache_prop(g, lv, "size")] if v.status == "ok"]
    s.append(f"### {n}.1 Cache size\n")
    s.append(img(figs["size"], readme, "Cache size per GPU and hierarchy level") + "\n")
    s.append(f"*Log₂ y-axis, because sizes range from {fmt_bytes(min(sizes))} to {fmt_bytes(max(sizes))}. "
             "Top: per-SM/CU caches. Bottom: device-level caches.*\n")
    s.append(values_toggle("cache size", gpus,
                           ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture", "l2", "l2seg", "l3"],
                           lambda g, lv: get_cache_prop(g, lv, "size"), fmt_bytes))
    s.append(f"### {n}.2 Cache line size\n")
    s.append(img(figs["line"], readme, "Cache line size per GPU and hierarchy level") + "\n")
    s.append(values_toggle("cache line size", gpus, LINE_LEVELS,
                           lambda g, lv: get_cache_prop(g, lv, "lineSize"), lambda v: f"{v:.0f} B"))
    s.append(f"### {n}.3 Fetch granularity\n")
    s.append(img(figs["fetch"], readme, "Fetch granularity per GPU and hierarchy level") + "\n")
    s.append("*The amount of data loaded per miss. It can be smaller than the line size, for example "
             "32 B sectors of a 128 B line.*\n")
    s.append(values_toggle("fetch granularity", gpus, FETCH_LEVELS,
                           lambda g, lv: get_cache_prop(g, lv, "fetchGranularity"), lambda v: f"{v:.0f} B"))

    if plans:
        s.append(head("Bandwidth"))
        bw_readme(s, num["Bandwidth"], plans, runs, readme)
    readme.write_text("\n".join(s))
    print(f"  wrote {readme.relative_to(REPO_ROOT) if readme.is_relative_to(REPO_ROOT) else readme}")


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(readme: Path, written: list[Path], plans: list[BwPlot], runs: list[Run], plotted: int) -> None:
    text = readme.read_text()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) + re.findall(r'<img src="([^"]+)"', text)
    for link in links:
        if not (readme.parent / link).is_file():
            raise SystemExit(f"error: README embeds missing image '{link}'")
    embedded = {(readme.parent / l).resolve() for l in links}
    for p in written + [pl.path for pl in plans]:
        if p.resolve() not in embedded:
            raise SystemExit(f"error: figure '{p}' is not embedded in the README")
    total = sum(len(r.measurements) for r in runs)
    in_plans = sum(pl.n_points for pl in plans)
    if not (total == in_plans == plotted):
        raise SystemExit(f"error: {total} valid measurements, {in_plans} assigned to plots, {plotted} drawn")
    if len({pl.path for pl in plans}) != len(plans):
        raise SystemExit("error: two bandwidth plots share a file name")
    print(f"validated: {len(links)} embedded figures ({len(plans)} bandwidth), {plotted} bandwidth "
          f"measurements drawn once each")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "sample_results",
                    help="directory with one MT4G JSON per GPU (default: sample_results)")
    ap.add_argument("--outdir", type=Path, default=None, help="figure directory (default: <results-dir>/plots)")
    ap.add_argument("--readme", type=Path, default=None, help="README path (default: <results-dir>/README.md)")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    outdir = (args.outdir or results_dir / "plots").resolve()
    readme = (args.readme or results_dir / "README.md").resolve()

    jsons = sorted(results_dir.glob("*.json"))
    if not jsons:
        raise SystemExit(f"error: no MT4G result JSON files in '{results_dir}'")
    gpus = sorted((load_gpu_json(p) for p in jsons), key=Gpu.sort_key)
    print(f"{len(gpus)} GPU result files in {results_dir}")
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    figs: dict[str, Path] = {}
    plot_latency(gpus, outdir, written); figs["latency"] = written[-1]
    plot_cache_size(gpus, outdir, written); figs["size"] = written[-1]
    plot_bytes_bars(gpus, outdir, written, "lineSize", LINE_LEVELS, "cache_line_size_by_gpu",
                    "Cache line size by GPU and memory hierarchy level", "Cache line size (bytes)")
    figs["line"] = written[-1]
    plot_bytes_bars(gpus, outdir, written, "fetchGranularity", FETCH_LEVELS, "fetch_granularity_by_gpu",
                    "Fetch granularity by GPU and memory hierarchy level", "Fetch granularity (bytes)")
    figs["fetch"] = written[-1]

    for problem in check_sources():
        print(f"warning: bandwidth benchmark source missing, spec may be stale: {problem}")
    runs, warnings = load_runs(results_dir, exclude=outdir)
    for w in warnings:
        print(f"warning: {w}")
    print(f"{len(runs)} bandwidth run(s) with {sum(len(r.measurements) for r in runs)} valid measurements")
    plans, plotted = plot_bandwidth(bw_columns(gpus, runs), outdir)

    write_readme(readme, gpus, runs, figs, plans)
    validate(readme, written, plans, runs, plotted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
