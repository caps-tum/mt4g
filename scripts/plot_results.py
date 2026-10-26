#!/usr/bin/env python3
"""plot_results.py -- cross-GPU result plots and README for MT4G sample results.

Reads the MT4G result files, draws one set of comparison figures with the GPU
model on the x-axis, and writes ``sample_results/README.md``, which embeds every
figure and lists the underlying values.

Input: ``<results-dir>/*.json``, one MT4G result per GPU run (discovered, nothing
is hard-coded). Used for bandwidth, latency, cache size, cache line size and
fetch granularity.

Usage:
  scripts/plot_results.py
  scripts/plot_results.py --results-dir sample_results --outdir sample_results/plots

Requires matplotlib (see the plot venv notes in the main README).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory


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
# Eight hues plus one marker per level. Sub-levels (scalar L1, constant L1.5,  #
# L2 segment) reuse their family's hue with their own marker (hatch in bars).  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Level:
    key: str
    label: str
    color: str
    hatch: Optional[str]
    vendors: frozenset
    json_node: tuple  # path below result["memory"]
    marker: str = "o"  # point marker in the latency / bandwidth / size figures


BOTH = frozenset({"AMD", "NVIDIA"})
AMD = frozenset({"AMD"})
NVIDIA = frozenset({"NVIDIA"})

LEVELS: dict[str, Level] = {
    lv.key: lv
    for lv in [
        Level("shared", "Shared memory / LDS", "#eb6834", None, BOTH, ("shared",), "o"),
        Level("l1", "L1 (vector L1 on AMD)", "#2a78d6", None, BOTH, ("l1",), "s"),
        Level("scalarL1", "Scalar L1 (AMD)", "#2a78d6", "///", AMD, ("scalarL1",), "D"),
        Level("constL1", "Constant L1 (NVIDIA)", "#4a3aa7", None, NVIDIA, ("constant", "l1"), "^"),
        Level("constL15", "Constant L1.5 (NVIDIA)", "#4a3aa7", "///", NVIDIA, ("constant", "l1.5"), "v"),
        Level("readOnly", "Read-only cache (NVIDIA)", "#1baf7a", None, NVIDIA, ("readOnly",), "P"),
        Level("texture", "Texture cache (NVIDIA)", "#e87ba4", None, NVIDIA, ("texture",), "X"),
        Level("l2", "L2", "#eda100", None, BOTH, ("l2",), "h"),
        Level("l2seg", "L2 segment (NVIDIA)", "#eda100", "///", NVIDIA, ("l2",), "v"),
        Level("l3", "L3 / Infinity Cache (AMD)", "#008300", None, AMD, ("l3",), "*"),
        Level("main", "Main memory", "#e34948", None, BOTH, ("main",), "d"),
    ]
}


# --------------------------------------------------------------------------- #
# GPU records                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Gpu:
    name: str          # general.name, as reported by the runtime
    vendor: str
    cc: tuple
    memory: dict
    compute: dict
    source: Path
    hostname: str = ""  # meta.hostname (absent in older result files)
    clock_khz: Optional[float] = None  # general.clockRate in kHz

    @property
    def label(self) -> str:
        """x-axis label: name, hostname on a second line."""
        return f"{self.name}\n({self.hostname})" if self.hostname else self.name

    @property
    def display(self) -> str:
        """One-line form for tables."""
        return f"{self.name} ({self.hostname})" if self.hostname else self.name

    def sort_key(self) -> tuple:
        return (self.vendor, self.cc, self.name, self.hostname, self.source.name)


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


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
    clock = general.get("clockRate", {})
    clock_khz = None
    if isinstance(clock, dict) and _finite_positive(clock.get("value")):
        if clock.get("unit", "kHz") != "kHz":
            raise SystemExit(f"error: '{path}': general.clockRate unit {clock.get('unit')!r} is not kHz")
        clock_khz = float(clock["value"])
    return Gpu(
        name=general.get("name") or path.stem,
        vendor=vendor,
        cc=(cc.get("major", 0), cc.get("minor", 0)),
        memory=data["memory"],
        compute=data.get("compute", {}),
        source=path,
        hostname=data.get("meta", {}).get("hostname", ""),
        clock_khz=clock_khz,
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


def get_latency(gpu: Gpu, level: Level) -> Value:
    """Mean latency in ns: the measured mean cycles divided by the GPU's clockRate."""
    if not applicable(gpu, level):
        return Value("n/a")
    lat = node_of(gpu, level).get("latency")
    if not isinstance(lat, dict) or not _finite_positive(lat.get("mean")):
        return Value("missing")
    if lat.get("unit", "cycles") != "cycles":
        raise SystemExit(f"error: {gpu.source}: {level.key} latency unit {lat.get('unit')!r} is not cycles")
    if gpu.clock_khz is None:
        return Value("missing")
    # cycles / (kHz * 1e3 cycles/s) = seconds; * 1e9 -> ns
    return Value("ok", float(lat["mean"]) / gpu.clock_khz * 1e6)


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
    ax.set_xticklabels([g.label for g in gpus] if show_labels else [], rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=11)
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


def level_handles(levels: list[Level], with_reported: bool = False) -> list:
    hs = [Patch(facecolor=lv.color, edgecolor="white", hatch=lv.hatch, label=lv.label) for lv in levels]
    if with_reported:
        hs.append(Patch(facecolor="white", edgecolor="0.35", linewidth=1.2,
                        label="Hollow = reported by runtime/driver, not measured"))
    return hs


# --------------------------------------------------------------------------- #
# Grouped bars: GPU on x, one bar per applicable level                         #
# --------------------------------------------------------------------------- #
def grouped_bars(ax, gpus: list[Gpu], levels: list[Level], getter: Callable[[Gpu, Level], Value],
                 value_fmt: Callable[[float], str] = lambda v: f"{v:.0f}", annotate: bool = True) -> set:
    """Draw one bar per level that applies to each GPU's vendor.

    Missing / low-confidence values are left out (no zero-height bar).
    Returns the levels that were actually drawn.
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
    return drawn


# --------------------------------------------------------------------------- #
# Point panels: GPU on x, one marker per level, all on the GPU's own line      #
# --------------------------------------------------------------------------- #
# Every value of a GPU sits exactly on that GPU's x position. Markers are open
# (thick coloured outline, no fill) with one shape per level, so values that
# coincide stay readable as overlaid shapes. Missing values are simply not drawn.
MARKER_SIZE = {"*": 260, "p": 150, "P": 150, "X": 150}  # shapes that look small at the default area


def point_panel(ax, gpus: list, levels: list, getter: Callable,
                setup: Optional[Callable] = None) -> tuple[set, list, bool]:
    """Draw the points; ``setup(ax, values)`` sets the scale and limits first, because
    overlap is judged on screen. Points of one GPU that would overlap are drawn
    concentric with decreasing size, so every outline stays visible.
    Returns the drawn level keys, all plotted values and whether any was runtime-reported."""
    pts = [(i, lv, v) for i, g in enumerate(gpus) for lv in levels
           for v in [getter(g, lv)] if v.status == "ok"]
    vals = [v.value for _, _, v in pts]
    (setup or linear_ylim)(ax, vals)
    ax.set_xlim(-0.6, len(gpus) - 0.4)
    ax.figure.canvas.draw()  # finalise the transform before measuring on-screen distances
    px = lambda val: ax.transData.transform((0, val))[1]
    base = lambda lv: MARKER_SIZE.get(lv.marker, 120)
    drawn, reported_any = set(), False
    for i in range(len(gpus)):
        mine = sorted((p for p in pts if p[0] == i), key=lambda p: px(p[2].value))
        clusters, cur = [], []
        for p in mine:  # neighbours closer than one marker diameter overlap
            if cur and px(p[2].value) - px(cur[-1][2].value) > 0.8 * math.sqrt(base(p[1])) * ax.figure.dpi / 72:
                clusters.append(cur)
                cur = []
            cur.append(p)
        if cur:
            clusters.append(cur)
        for cl in clusters:
            for rank, (_, lv, v) in enumerate(cl):  # first = biggest, drawn underneath
                grow = 1 + 0.9 * (len(cl) - 1 - rank)
                reported = v.source == "reported"
                reported_any |= reported
                ax.scatter([i], [v.value], s=base(lv) * grow, marker=lv.marker,
                           facecolor=to_rgba(lv.color, 0.3) if reported else "none", edgecolor=lv.color,
                           linewidth=2.2, zorder=3 + rank)
                drawn.add(lv.key)
    return drawn, vals, reported_any


def point_handles(levels: list, drawn: set, reported: bool = False) -> list:
    hs = [Line2D([], [], linestyle="none", marker=lv.marker, markersize=15 if lv.marker == "*" else 11,
                 markerfacecolor="none", markeredgecolor=lv.color, markeredgewidth=2.2, label=lv.label)
          for lv in levels if lv.key in drawn]
    if reported:
        hs.append(Line2D([], [], linestyle="none", marker="o", markersize=11, markerfacecolor=to_rgba("0.35", 0.3),
                         markeredgecolor="0.35", markeredgewidth=2.2,
                         label="Light fill = reported by runtime/driver"))
    return hs


def linear_ylim(ax, vals: list) -> None:
    ax.set_ylim(0, (max(vals) if vals else 1) * 1.12)


# --------------------------------------------------------------------------- #
# Latency                                                                      #
# --------------------------------------------------------------------------- #
LAT_PANELS = [
    ("SM/CU-local memory", ["shared", "l1", "scalarL1", "constL1", "constL15", "readOnly", "texture"]),
    ("L2, L3 and main memory", ["l2", "l3", "main"]),
]


def plot_latency(gpus: list[Gpu], outdir: Path, written: list[Path]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(17, 14))
    for ax, (title, keys) in zip(axes, LAT_PANELS):
        levels = [LEVELS[k] for k in keys]
        drawn, _, _ = point_panel(ax, gpus, levels, get_latency)
        ax.set_ylabel("Mean latency (ns)")
        ax.set_title(title, loc="left", pad=16)
        style_gpu_axis(ax, gpus)
        ax.set_xlabel("GPU")
        ax.legend(handles=point_handles(levels, drawn), loc="upper left", bbox_to_anchor=(1.005, 1.0),
                  frameon=False)
    fig.suptitle("Memory access latency by GPU and memory hierarchy level (mean, ns)", x=0.01, ha="left",
                 fontsize=16.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "latency_mean_by_gpu", written)


# --------------------------------------------------------------------------- #
# Cache properties                                                             #
# --------------------------------------------------------------------------- #
SIZE_PANELS = [
    ("SM/CU-local caches (measured)", ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture"]),
    ("Device-level caches", ["l2", "l2seg", "l3"]),
]
LINE_LEVELS = ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture", "l2", "l3"]
FETCH_LEVELS = ["l1", "scalarL1", "constL1", "constL15", "readOnly", "texture", "l2"]


def log2_bytes_axis(ax, vals: list) -> None:
    ax.set_yscale("log", base=2)
    lo = 2 ** math.floor(math.log2(min(vals)) - 0.5)
    hi = 2 ** math.ceil(math.log2(max(vals)) + 0.7)
    ax.set_ylim(lo, hi)
    ticks = [2 ** e for e in range(int(math.log2(lo)), int(math.log2(hi)) + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt_bytes(t) for t in ticks])
    ax.minorticks_off()


def plot_cache_size(gpus: list[Gpu], outdir: Path, written: list[Path]) -> None:
    """Dot plot on a log2 axis: sizes span 2 KiB .. 256 MiB."""
    fig, axes = plt.subplots(2, 1, figsize=(17, 14))
    for ax, (title, keys) in zip(axes, SIZE_PANELS):
        levels = [LEVELS[k] for k in keys]
        drawn, _, any_reported = point_panel(ax, gpus, levels, lambda g, lv: get_cache_prop(g, lv, "size"),
                                             setup=log2_bytes_axis)
        ax.set_ylabel("Cache size (bytes, log₂ scale)")
        ax.set_title(title, loc="left", pad=16)
        style_gpu_axis(ax, gpus)
        hs = point_handles(levels, drawn, any_reported)
        ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    for ax in axes:
        ax.set_xlabel("GPU")
    fig.suptitle("Cache size by GPU and memory hierarchy level (log₂ y-axis)", x=0.01, ha="left",
                 fontsize=16.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "cache_size_by_gpu", written)


def plot_bytes_bars(gpus: list[Gpu], outdir: Path, written: list[Path], metric: str, keys: list[str],
                    name: str, title: str, ylabel: str) -> None:
    levels = [LEVELS[k] for k in keys]
    getter = lambda g, lv: get_cache_prop(g, lv, metric)
    fig, ax = plt.subplots(figsize=(17, 7.6))
    drawn = grouped_bars(ax, gpus, levels, getter)
    vals = [getter(g, lv).value for g in gpus for lv in levels if getter(g, lv).status == "ok"]
    ymax = max(vals)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_yticks(range(0, int(ymax) + 1, 32))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("GPU")
    style_gpu_axis(ax, gpus)
    any_reported = any(getter(g, lv).source == "reported" and getter(g, lv).status == "ok"
                       for g in gpus for lv in levels)
    ax.legend(handles=level_handles([lv for lv in levels if lv.key in drawn], with_reported=any_reported),
              loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.set_title(title, loc="left", pad=18, fontsize=16.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, name, written)


# --------------------------------------------------------------------------- #
# Bandwidth (from the MT4G JSON)                                               #
# --------------------------------------------------------------------------- #
# Each bandwidth benchmark serialises a CacheBandwidthResult
# (src/typedef/cacheBandwidthResult.hpp): measuredBandwidth in GiB/s for the best
# swept configuration, plus numBlocks / numThreads / numReps. main.cpp stores it
# under memory.<level>.<key>. SM/CU-local levels run one block on one SM/CU and
# report per-SM/CU bandwidth (key *PerCU); L2, L3 and main memory are device-wide.
@dataclass(frozen=True)
class BwSeries:
    key: str
    label: str
    color: str
    marker: str
    vendors: frozenset
    json_node: tuple   # below result["memory"]
    read_key: Optional[str]
    write_key: Optional[str]


def _bw(level: str, label: str, read_key: Optional[str], write_key: Optional[str], key: str = "",
        marker: str = "") -> BwSeries:
    lv = LEVELS[level]
    return BwSeries(key or level, label, lv.color, marker or lv.marker, lv.vendors, lv.json_node,
                    read_key, write_key)


BW_SERIES = [
    _bw("shared", "Shared memory / LDS, dynamic (SM/CU-local)", "readBandwidthPerCU", "writeBandwidthPerCU"),
    _bw("shared", "Shared memory / LDS, static (SM/CU-local)", "readBandwidthPerCUStatic",
        "writeBandwidthPerCUStatic", key="sharedStatic", marker="p"),
    _bw("l1", "L1 (SM/CU-local)", "readBandwidthPerCU", "writeBandwidthPerCU"),
    _bw("scalarL1", "Scalar L1 (AMD, SM/CU-local)", "readBandwidthPerCU", "writeBandwidthPerCU"),
    _bw("constL1", "Constant L1 (NVIDIA, SM/CU-local)", "readBandwidthPerCU", None),
    _bw("constL15", "Constant L1.5 (NVIDIA, SM/CU-local)", "readBandwidthPerCU", None),
    _bw("readOnly", "Read-only cache (NVIDIA, SM/CU-local)", "readBandwidthPerCU", None),
    _bw("texture", "Texture cache (NVIDIA, SM/CU-local)", "readBandwidthPerCU", None),
    _bw("l2", "L2 (device-wide)", "readBandwidth", "writeBandwidth"),
    _bw("l3", "L3 / Infinity Cache (AMD, device-wide)", "readBandwidth", "writeBandwidth"),
    _bw("main", "Main memory (device-wide)", "readBandwidth", "writeBandwidth"),
]


@dataclass
class BwValue(Value):
    blocks: Optional[int] = None
    threads: Optional[int] = None


def get_bandwidth(gpu: Gpu, series: BwSeries, direction: str) -> BwValue:
    """Measured bandwidth in GiB/s. Only the current CacheBandwidthResult format is
    used; the older scalar ``{"value": ...}`` entries are treated as missing."""
    key = series.read_key if direction == "read" else series.write_key
    if key is None or gpu.vendor not in series.vendors or (series.key == "l3" and "l3" not in gpu.memory):
        return BwValue("n/a")
    node = gpu.memory
    for part in series.json_node:
        node = node.get(part, {}) if isinstance(node, dict) else {}
    entry = node.get(key) if isinstance(node, dict) else None
    if not isinstance(entry, dict) or not _finite_positive(entry.get("measuredBandwidth")):
        return BwValue("missing")
    return BwValue("ok", float(entry["measuredBandwidth"]), blocks=entry.get("numBlocks"),
                   threads=entry.get("numThreads"))


def plot_bandwidth(gpus: list[Gpu], direction: str, outdir: Path, written: list[Path]) -> None:
    series = [sr for sr in BW_SERIES if (sr.read_key if direction == "read" else sr.write_key)]
    getter = lambda g, sr: get_bandwidth(g, sr, direction)
    fig, ax = plt.subplots(figsize=(17, 8))
    def log_axis(ax, vals):
        if vals:
            ax.set_yscale("log")
            ax.set_ylim(min(vals) / 3, max(vals) * 3)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 1 else f"{y:g}"))
    drawn, vals, _ = point_panel(ax, gpus, series, getter, setup=log_axis)
    if not vals:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(0.5, 0.55, "No measurements in the current format yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=20, color="0.7", fontweight="bold")
    ax.set_ylabel("Measured bandwidth (GiB/s, log scale)")
    ax.set_xlabel("GPU")
    style_gpu_axis(ax, gpus)
    ax.legend(handles=point_handles(series, drawn),
              loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.set_title(f"{direction.capitalize()} bandwidth by GPU and memory hierarchy level", loc="left", pad=24,
                 fontsize=16.5, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, f"bandwidth_{direction}_by_gpu", written)


def bw_values_toggle(gpus: list[Gpu], direction: str) -> str:
    series = [sr for sr in BW_SERIES if (sr.read_key if direction == "read" else sr.write_key)]
    series = [sr for sr in series if any(get_bandwidth(g, sr, direction).status != "n/a" for g in gpus)]

    def fmt(v: BwValue) -> str:
        if v.status == "n/a":
            return ""
        if v.status != "ok":
            return "—"
        return fmt_gib(v.value)

    rows = [[g.display] + [fmt(get_bandwidth(g, sr, direction)) for sr in series] for g in gpus]
    return (f"<details><summary>Values ({direction} bandwidth, GiB/s; — = no valid value; blank = not applicable)</summary>\n\n"
            + md_table(["GPU"] + [sr.label for sr in series], rows) + "\n\n</details>\n")


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
    rows = [[g.display] + [cell(getter(g, lv), fmt) for lv in levels] for g in gpus]
    return md_table(["GPU"] + [lv.label for lv in levels], rows)


REPORTED_NOTE = ("Lightly filled markers, hollow bars and † in the tables mark values reported by the HIP "
                 "runtime rather than measured.")


def values_toggle(metric: str, gpus: list[Gpu], keys: list[str], getter, fmt) -> str:
    """Collapsible value table; the runtime-reported note is added only if the table has such values."""
    reported = any(getter(g, LEVELS[k]).status == "ok" and getter(g, LEVELS[k]).source == "reported"
                   for g in gpus for k in keys)
    note = f"{REPORTED_NOTE}\n\n" if reported else ""
    return (f"<details><summary>Values ({metric}; — = no valid value; blank = not applicable)</summary>\n\n"
            + note + metric_table(gpus, keys, getter, fmt) + "\n\n</details>\n")


def img(path: Path, readme: Path, alt: str) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, readme.parent)).as_posix()})"


def write_readme(readme: Path, gpus: list[Gpu], figs: dict[str, Path]) -> None:
    rel = lambda p: Path(os.path.relpath(p, readme.parent)).as_posix()
    n_amd = sum(g.vendor == "AMD" for g in gpus)
    sections = ["Bandwidth", "Memory latency", "Cache properties", "Overview"]
    num = {name: i + 1 for i, name in enumerate(sections)}
    head = lambda name: f"## {num[name]}. {name}\n"
    anchor = lambda name: f"#{num[name]}-{name.lower().replace(' ', '-')}"

    s = ["# MT4G Results: GPU Memory Hierarchy Across AMD and NVIDIA GPUs\n"]
    s.append("## Contents\n\n" + "\n".join(f"{num[n]}. [{n}]({anchor(n)})" for n in sections) + "\n")

    s.append(head("Bandwidth"))
    s.append("SM/CU-local levels run a single block on one SM/CU and report per-SM/CU bandwidth; L2, L3 and main memory "
             "are device-wide. Log y-axis, because the values span several orders of magnitude.\n")
    n = num["Bandwidth"]
    for i, d in enumerate(("read", "write"), 1):
        s.append(f"### {n}.{i} {d.capitalize()} bandwidth\n")
        s.append(img(figs[f"bw_{d}"], readme, f"{d.capitalize()} bandwidth per GPU and hierarchy level") + "\n")
        s.append(bw_values_toggle(gpus, d))

    s.append(head("Memory latency"))
    s.append(img(figs["latency"], readme, "Mean memory latency per GPU and hierarchy level") + "\n")
    s.append("*Mean latency in ns: the measured mean in cycles divided by the GPU's `clockRate` from the same "
             "JSON.*\n")
    s.append(values_toggle("mean latency, ns", gpus, [k for k in LEVELS if k != "l2seg"], get_latency,
                           lambda v: f"{v:.1f}"))

    s.append(head("Cache properties"))
    n = num["Cache properties"]
    sizes = [v.value for g in gpus for lv in LEVELS.values()
             for v in [get_cache_prop(g, lv, "size")] if v.status == "ok"]
    s.append(f"### {n}.1 Cache size\n")
    s.append(img(figs["size"], readme, "Cache size per GPU and hierarchy level") + "\n")
    s.append(f"*Log₂ y-axis, because sizes range from {fmt_bytes(min(sizes))} to {fmt_bytes(max(sizes))}. "
             "Top: SM/CU-local caches. Bottom: device-level caches.*\n")
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

    s.append(head("Overview"))
    s.append(f"One MT4G result file per GPU run: {len(gpus)} files, {len(gpus) - n_amd} NVIDIA and {n_amd} AMD.\n")
    s.append(md_table(["GPU", "Host", "Vendor", "Compute capability (HIP)", "SMs/CUs", "Clock (MHz)", "Result file"],
                      [[g.name, g.hostname or "—", g.vendor, f"{g.cc[0]}.{g.cc[1]}",
                        str(g.compute.get("multiProcessorCount", "?")),
                        f"{g.clock_khz / 1000:.0f}" if g.clock_khz else "—",
                        f"[`{g.source.name}`]({rel(g.source).replace(' ', '%20')})"] for g in gpus]) + "\n")
    s.append("Metrics covered:\n\n" + "\n".join(f"- {m}" for m in [
        "Read and write bandwidth, in GiB/s", "Memory latency: mean, in ns", "Cache size, in bytes",
        "Cache line size, in bytes", "Fetch granularity, in bytes"]) + "\n")
    s.append("This page is generated by `scripts/plot_results.py`; edit the script, not this file. Re-run it "
             "after adding result files (needs Python 3.9+ with matplotlib):\n\n```bash\n"
             "python3 scripts/plot_results.py\n```\n")
    readme.write_text("\n".join(s))
    print(f"  wrote {readme.relative_to(REPO_ROOT) if readme.is_relative_to(REPO_ROOT) else readme}")


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(readme: Path, written: list[Path]) -> None:
    text = readme.read_text()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    for link in links:
        if not (readme.parent / link).is_file():
            raise SystemExit(f"error: README embeds missing image '{link}'")
    embedded = {(readme.parent / l).resolve() for l in links}
    for p in written:
        if p.resolve() not in embedded:
            raise SystemExit(f"error: figure '{p}' is not embedded in the README")
    print(f"validated: {len(links)} embedded figures, all present")


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
    for d in ("read", "write"):
        plot_bandwidth(gpus, d, outdir, written); figs[f"bw_{d}"] = written[-1]
    plot_latency(gpus, outdir, written); figs["latency"] = written[-1]
    plot_cache_size(gpus, outdir, written); figs["size"] = written[-1]
    plot_bytes_bars(gpus, outdir, written, "lineSize", LINE_LEVELS, "cache_line_size_by_gpu",
                    "Cache line size by GPU and memory hierarchy level", "Cache line size (bytes)")
    figs["line"] = written[-1]
    plot_bytes_bars(gpus, outdir, written, "fetchGranularity", FETCH_LEVELS, "fetch_granularity_by_gpu",
                    "Fetch granularity by GPU and memory hierarchy level", "Fetch granularity (bytes)")
    figs["fetch"] = written[-1]

    write_readme(readme, gpus, figs)
    validate(readme, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
