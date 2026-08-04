#!/usr/bin/env python3
"""plot_bandwidth.py -- bandwidth plots for MT4G.

Single source of truth for bandwidth visualization.

All benchmarks now use a fixed repetition count (MIN_REPS == MAX_REPS),
so plots show bandwidth vs thread count only.

Supported plot types:
  1. Block-sweep benchmarks (L2 / L3 / vL1d / sL1d) -- peak bandwidth per block
     size, with every measured (thread-count) point scattered and annotated.
     The thread sweep stops early once bandwidth drops, so the dataset is
     truncated per block and only the peak is comparable across blocks.
  2. Single-line benchmarks (L1)
  3. LDS/shared memory (allocation-based grouping)

Input:
  CSV grid files produced by the MT4G bandwidth benchmarks.

Usage examples:
  plot_bandwidth.py blocksweep --input file.csv --outdir out/
  plot_bandwidth.py l1 --input file.csv --outdir out/
  plot_bandwidth.py auto --indir sample_results/ --outdir sample_results/

Outputs:
  One PNG per benchmark figure (overwritten by default).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------- #
# Global publication style (clean white background, visible grid, readable     #
# fonts).  Kept in one place so every figure looks consistent.                 #
# --------------------------------------------------------------------------- #
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,  # grid behind the data lines
        "lines.linewidth": 1.8,
        "lines.marker": "o",
        "lines.markersize": 4,
    }
)


def _apply_log2_thread_axis(ax, threads: list[int]) -> None:
    """Configure a log2 x-axis labelled with the actual thread counts.

    Thread counts are powers of two (e.g. 32, 64, ..., 1024); a log2 scale gives
    them equal spacing.  Unlike the old repetition axis we label every tick with
    the integer thread count (there are only a handful of them).
    """
    vals = sorted({t for t in threads if t > 0})
    if not vals:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    ax.set_xticklabels([str(v) for v in vals])
    ax.xaxis.set_minor_locator(mticker.NullLocator())


# --------------------------------------------------------------------------- #
# Strongly typed in-memory representation of a parsed grid CSV.                 #
# --------------------------------------------------------------------------- #
@dataclass
class BlockSweepGrid:
    """A block grid: bandwidth[block][thread][rep] in GiB/s.

    With the single hard-coded repetition count ``rep`` is degenerate (one
    entry); :meth:`collapse` reduces it to bandwidth[block][thread].
    """

    blocks: list[int]
    threads: list[int]
    reps: list[int]
    # values[b][t][r] indexed the same way as blocks/threads/reps
    values: list[list[list[float]]]

    def collapse(self) -> list[list[float]]:
        """Reduce the repetition axis -> bandwidth[block][thread].

        No assumption is made about how many repetition columns exist: the best
        (max) bandwidth over whatever repetitions are present is taken, which is
        exactly the single measured value when only one repetition is recorded.
        """
        out: list[list[float]] = []
        for b_vals in self.values:
            row = []
            for t_vals in b_vals:
                finite = [v for v in t_vals if not math.isnan(v)]
                row.append(max(finite) if finite else float("nan"))
            out.append(row)
        return out


@dataclass
class ConfigGrid:
    """A 2D grid for a single configuration: bandwidth[thread][rep] in GiB/s."""

    threads: list[int]
    reps: list[int]
    values: list[list[float]]  # values[t][r]
    size_kib: Optional[int] = None       # array size in KiB (from filename)
    alloc: Optional[str] = None          # 'dyn' | 'stat' (from filename)
    direction: str = "read"              # 'read' | 'write'
    source: str = ""                     # originating file (for diagnostics)

    def collapse(self) -> list[float]:
        """Reduce the repetition axis -> bandwidth[thread] (max over reps)."""
        out: list[float] = []
        for row in self.values:
            finite = [v for v in row if not math.isnan(v)]
            out.append(max(finite) if finite else float("nan"))
        return out

    def alloc_label(self) -> str:
        """Legend label grouped by allocation type, e.g. 'dynamic (24 KiB)'."""
        names = {"dyn": "dynamic", "stat": "static"}
        alloc = names.get(self.alloc or "", self.alloc or "?")
        if self.size_kib is not None:
            return f"{alloc} ({self.size_kib} KiB)"
        return alloc

    def sort_key(self) -> tuple:
        """Order configs as dynamic before static, then by size."""
        return (0 if self.alloc == "dyn" else 1, self.size_kib or 0)


# --------------------------------------------------------------------------- #
# CSV loading                                                                  #
# --------------------------------------------------------------------------- #
def _is_block_sweep(header: str) -> bool:
    return header.strip().lower().startswith("blocks,")


def load_block_sweep(path: Path) -> BlockSweepGrid:
    """Parse a long-format ``blocks,threads,reps,bandwidth`` CSV."""
    blocks: list[int] = []
    threads: list[int] = []
    reps: list[int] = []
    # nested dict keeps insertion-free lookup while we discover the axes
    table: dict[tuple[int, int, int], float] = {}
    with path.open() as fh:
        next(fh, None)  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            b, t, r, v = line.split(",")
            b, t, r, v = int(b), int(t), int(r), float(v)
            if b not in blocks:
                blocks.append(b)
            if t not in threads:
                threads.append(t)
            if r not in reps:
                reps.append(r)
            table[(b, t, r)] = v
    blocks.sort()
    threads.sort()
    reps.sort()
    values = [[[table.get((b, t, r), float("nan")) for r in reps] for t in threads] for b in blocks]
    return BlockSweepGrid(blocks, threads, reps, values)


def load_config_grid(path: Path) -> ConfigGrid:
    """Parse a wide-format ``threads,<rep>[,<rep>...]`` CSV (single configuration)."""
    with path.open() as fh:
        header = fh.readline().strip().split(",")
        reps = [int(x) for x in header[1:]]
        threads: list[int] = []
        values: list[list[float]] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            threads.append(int(parts[0]))
            values.append([float(x) for x in parts[1:]])
    cfg = ConfigGrid(threads=threads, reps=reps, values=values, source=str(path))
    _annotate_from_filename(cfg, path.name)
    return cfg


def _annotate_from_filename(cfg: ConfigGrid, name: str) -> None:
    """Recover (array size, allocation, direction) metadata from the file name.

    The C++ exporter encodes these for LDS grids, e.g.
    ``MI300A__LDS_Read_16KiB_dyn_BW_Grid.csv``.  Missing pieces stay ``None``.
    """
    low = name.lower()
    cfg.direction = "write" if "write" in low else "read"
    m = re.search(r"(\d+)\s*kib", low)
    if m:
        cfg.size_kib = int(m.group(1))
    if "stat" in low:
        cfg.alloc = "stat"
    elif "dyn" in low:
        cfg.alloc = "dyn"


# --------------------------------------------------------------------------- #
# Core figure: bandwidth vs threads, one line per series                       #
# --------------------------------------------------------------------------- #
def plot_bw_vs_threads(
    series: list[tuple[Optional[str], list[int], list[float]]],
    title: str,
    outdir: Path,
    outbase: str,
    dpi: int,
    legend_title: Optional[str] = None,
) -> None:
    """Plot bandwidth (y) against thread count (x), one line per series.

    ``series`` is a list of ``(label, threads, values)`` tuples.  A single,
    unlabelled series is drawn as a lone line with no legend (e.g. NVIDIA L1 or
    any benchmark with only one block count).
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    all_threads: list[int] = []
    ymax = 0.0
    labelled = False
    for label, threads, values in series:
        ax.plot(threads, values, label=label)
        if label is not None:
            labelled = True
        all_threads = threads if len(threads) > len(all_threads) else all_threads
        finite = [v for v in values if not math.isnan(v)]
        if finite:
            ymax = max(ymax, max(finite))

    _apply_log2_thread_axis(ax, all_threads)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_title(title)
    # Explicit y-limits with 15% headroom so peak markers never touch the frame.
    ax.set_ylim(0, (ymax if ymax > 0 else 1.0) * 1.15)

    if labelled:
        ncol = 2 if len(series) > 6 else 1
        ax.legend(title=legend_title, ncol=ncol, loc="best", framealpha=0.9)

    fig.tight_layout()
    _save(fig, outdir, outbase, dpi)


# --------------------------------------------------------------------------- #
# Builders that turn parsed grids into series for plot_bw_vs_threads           #
# --------------------------------------------------------------------------- #
# Colours for the peak figure: a muted blue for the raw measurements and a
# strong red for the headline peak markers/line.
_MEASURED_COLOR = "#4C72B0"
_PEAK_COLOR = "#C44E52"


def plot_block_peak(grid: BlockSweepGrid, title: str, outdir: Path, outbase: str, dpi: int) -> None:
    """Peak bandwidth by block size.

    The thread sweep for each block stops once bandwidth declines, so blocks are
    measured with different thread counts. This plot shows the peak bandwidth for
    each block (connected by a line), all measured points, and the corresponding
    thread count annotated beside each point (highlighted for the peak). Block
    sizes use evenly spaced categorical x-positions.
    """
    collapsed = grid.collapse()  # collapsed[b][t], NaN where never measured
    blocks = grid.blocks
    threads = grid.threads
    xpos = list(range(len(blocks)))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    peak_vals: list[float] = []
    scatter_labelled = False
    for x, bi in enumerate(range(len(blocks))):
        measured = [
            (threads[ti], collapsed[bi][ti])
            for ti in range(len(threads))
            if not math.isnan(collapsed[bi][ti])
        ]
        if not measured:
            peak_vals.append(float("nan"))
            continue

        peak = max(v for _, v in measured)
        peak_vals.append(peak)

        for t, v in measured:
            ax.scatter(
                [x], [v], s=30, color=_MEASURED_COLOR, zorder=3,
                edgecolors="white", linewidths=0.5,
                label=None if scatter_labelled else "Measured",
            )
            scatter_labelled = True
            if v == peak:
                ax.annotate(
                    str(t), (x, v),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", va="bottom", fontsize=8,
                    color=_PEAK_COLOR, fontweight="bold",
                )
            else:
                ax.annotate(
                    str(t), (x, v),
                    textcoords="offset points", xytext=(6, 0),
                    ha="left", va="center", fontsize=8, color="0.35",
                )

    ax.plot(
        xpos, peak_vals,
        color=_PEAK_COLOR, linewidth=1.8, marker="D", markersize=8,
        markerfacecolor=_PEAK_COLOR, markeredgecolor="white", markeredgewidth=0.8,
        zorder=4, label="Peak bandwidth",
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Blocks")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_title(title)

    finite_peaks = [v for v in peak_vals if not math.isnan(v)]
    ymax = max(finite_peaks) if finite_peaks else 1.0
    ax.set_ylim(0, ymax * 1.15)
    # Padding so the thread-count labels near the frame stay readable.
    ax.set_xlim(-0.5, len(blocks) - 0.5 + 0.4)

    ax.legend(loc="best", framealpha=0.9, title="Number by each point = threads/block")
    fig.tight_layout()
    _save(fig, outdir, outbase, dpi)


def plot_single_line(grid: ConfigGrid, title: str, outdir: Path, outbase: str, dpi: int) -> None:
    """Single-block benchmark (e.g. NVIDIA L1): one line, x = threads."""
    series = [(None, grid.threads, grid.collapse())]
    plot_bw_vs_threads(series, title, outdir, outbase, dpi)


def plot_lds_alloc_lines(
    configs: list[ConfigGrid], gpu: str, direction: str, outdir: Path, outbase: str, dpi: int
) -> None:
    """LDS: one line per allocation type (dynamic vs. static), x = threads."""
    configs = sorted(configs, key=ConfigGrid.sort_key)
    series = [(cfg.alloc_label(), cfg.threads, cfg.collapse()) for cfg in configs]
    title = f"{gpu}: LDS {direction} bandwidth"
    plot_bw_vs_threads(series, title, outdir, outbase, dpi, legend_title="Allocation")


# --------------------------------------------------------------------------- #
# Saving / naming helpers                                                      #
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    """Turn a human title into a safe file stem (mirrors the existing charts)."""
    text = text.replace(" - ", "__")
    text = re.sub(r"[^\w.-]+", "_", text)
    return text.strip("_")


def _save(fig, outdir: Path, outbase: str, dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{outbase}.png"
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"[plot_bandwidth] wrote {png}")


# --------------------------------------------------------------------------- #
# "auto" mode: classify every grid CSV in a directory and plot all figures     #
# --------------------------------------------------------------------------- #
# Maps the benchmark token found in a grid file name to a thesis-style label.
# The C++ exporter writes the display token directly (vL1d/sL1d on AMD, L1 on
# NVIDIA) so the figure titles are vendor-correct without inspecting the data.
_PRETTY = {
    "vl1d": "vL1d",
    "sl1d": "sL1d",
    "mainmemory": "Main Memory",
    "readonly": "Read-Only",
    "texture": "Texture",
    # Listed before the plain "l1" entry so a constant grid never matches it.
    "constantl1.5": "Constant L1.5",
    "constantl1": "Constant L1",
    "l1": "L1",
    "l2": "L2",
    "l3": "L3",
    "lds": "LDS",
}


def _benchmark_token(name: str) -> str:
    low = name.lower()
    for token, pretty in _PRETTY.items():
        if f"__{token}_" in low or f"_{token}_bw" in low:
            return pretty
    return "Bandwidth"


# Block-sweep benchmarks (3D grids). vL1d/sL1d are AMD; L2/L3/Main Memory work
# on both vendors.
_BLOCK_SWEEP_TOKENS = {"vL1d", "sL1d", "L2", "L3", "Main Memory"}

# Single-block (per-SM) sweeps: one line per thread count, using the shared
# 2D (threads x reps) grid for NVIDIA cache read-bandwidth benchmarks.
_SINGLE_LINE_TOKENS = {"L1", "Read-Only", "Texture", "Constant L1", "Constant L1.5"}


def _benchmark_category(name: str) -> tuple[str, str]:
    """Classify a grid file by *benchmark type* (filename token).

    Returns ``(category, pretty_token)`` where category is one of:
        "block_sweep" -> L2 / L3 / vL1d / sL1d          (block lines, x=threads)
        "single_line" -> NVIDIA L1 / Read-Only / Texture (single line, x=threads)
        "lds"         -> LDS / shared memory             (allocation lines, x=threads)
        "unknown"     -> fall back to grid dimensionality
    """
    token = _benchmark_token(name)
    if token in _BLOCK_SWEEP_TOKENS:
        return "block_sweep", token
    if token in _SINGLE_LINE_TOKENS:
        return "single_line", token
    if token == "LDS":
        return "lds", token
    return "unknown", token


def _gpu_from_name(name: str) -> str:
    return name.split("__", 1)[0] if "__" in name else "GPU"


def run_auto(indir: Path, outdir: Path, dpi: int) -> int:
    csvs = sorted(indir.glob("*_BW_Grid.csv")) + sorted(indir.glob("*BW_Grid*.csv"))
    csvs = sorted(set(csvs))
    if not csvs:
        print(f"[plot_bandwidth] no '*BW_Grid*.csv' files found in {indir}", file=sys.stderr)
        return 1

    lds_configs: dict[str, list[ConfigGrid]] = {"read": [], "write": []}
    n_figs = 0
    for csv in csvs:
        header = csv.open().readline()
        low = csv.name.lower()
        direction = "write" if "write" in low else "read"
        gpu = _gpu_from_name(csv.name)
        category, token = _benchmark_category(csv.name)

        if category == "block_sweep" or (category == "unknown" and _is_block_sweep(header)):
            grid = load_block_sweep(csv)
            title = f"{gpu}: {token} {direction} peak bandwidth per block"
            plot_block_peak(grid, title, outdir, _slug(f"{gpu} - {token} {direction} bandwidth"), dpi)
            n_figs += 1
        elif category == "single_line":
            grid = load_config_grid(csv)
            title = f"{gpu}: {token} {direction} bandwidth"
            plot_single_line(grid, title, outdir, _slug(f"{gpu} - {token} {direction} bandwidth"), dpi)
            n_figs += 1
        else:  # "lds" or unrecognised 2D grid -> group by allocation per direction
            cfg = load_config_grid(csv)
            lds_configs[cfg.direction].append(cfg)

    # Combine the LDS/shared configs into one allocation-lines figure per direction.
    for direction, configs in lds_configs.items():
        if not configs:
            continue
        gpu = _gpu_from_name(Path(configs[0].source).name)
        plot_lds_alloc_lines(configs, gpu, direction, outdir, _slug(f"{gpu} - LDS {direction} bandwidth"), dpi)
        n_figs += 1

    print(f"[plot_bandwidth] generated {n_figs} figure(s) into {outdir}")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    bs = sub.add_parser("blocksweep", help="peak-bandwidth-per-block figure (x=blocks) from a 3D grid CSV")
    bs.add_argument("--input", required=True, help="blocks,threads,reps,bandwidth CSV")
    bs.add_argument("--title", default="Bandwidth")
    bs.add_argument("--outdir", default=".")
    bs.add_argument("--outfile", default=None, help="output file stem (no extension)")
    bs.add_argument("--dpi", type=int, default=150)

    ld = sub.add_parser("lds", help="LDS allocation-lines figure (x=threads) from one or more 2D grid CSVs")
    ld.add_argument("--input", action="append", required=True, help="threads,<rep>... CSV (repeatable)")
    ld.add_argument("--gpu", default="GPU")
    ld.add_argument("--direction", default=None, choices=["read", "write"])
    ld.add_argument("--outdir", default=".")
    ld.add_argument("--outfile", default=None)
    ld.add_argument("--dpi", type=int, default=150)

    l1 = sub.add_parser("l1", help="single-line figure (x=threads) from one 2D grid CSV")
    l1.add_argument("--input", required=True, help="threads,<rep>... CSV")
    l1.add_argument("--title", default=None, help="figure title (default derived from filename)")
    l1.add_argument("--outdir", default=".")
    l1.add_argument("--outfile", default=None)
    l1.add_argument("--dpi", type=int, default=150)

    au = sub.add_parser("auto", help="scan a directory and regenerate every bandwidth figure")
    au.add_argument("--indir", required=True, help="directory containing *BW_Grid*.csv files")
    au.add_argument("--outdir", default=None, help="output directory (default: --indir)")
    au.add_argument("--dpi", type=int, default=150)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "blocksweep":
        grid = load_block_sweep(Path(args.input))
        outbase = args.outfile or _slug(args.title)
        plot_block_peak(grid, args.title, Path(args.outdir), outbase, args.dpi)
        return 0

    if args.mode == "lds":
        configs = [load_config_grid(Path(p)) for p in args.input]
        direction = args.direction or (configs[0].direction if configs else "read")
        outbase = args.outfile or _slug(f"{args.gpu} - LDS {direction} bandwidth")
        plot_lds_alloc_lines(configs, args.gpu, direction, Path(args.outdir), outbase, args.dpi)
        return 0

    if args.mode == "l1":
        path = Path(args.input)
        grid = load_config_grid(path)
        gpu = _gpu_from_name(path.name)
        title = args.title or f"{gpu}: L1 {grid.direction} bandwidth"
        outbase = args.outfile or _slug(f"{gpu} - L1 {grid.direction} bandwidth")
        plot_single_line(grid, title, Path(args.outdir), outbase, args.dpi)
        return 0

    if args.mode == "auto":
        indir = Path(args.indir)
        outdir = Path(args.outdir) if args.outdir else indir
        return run_auto(indir, outdir, args.dpi)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
