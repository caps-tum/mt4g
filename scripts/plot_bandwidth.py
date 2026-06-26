#!/usr/bin/env python3
"""plot_bandwidth.py -- bandwidth plots for MT4G.

This is the single source of truth for *all* bandwidth visualisation in MT4G.
It reproduces the two bandwidth figure styles used in the thesis
(`Nikolay_Romanenko_Abschlussarbeit.pdf`):

  1. BLOCK-SWEEP GRID  (vL1d / sL1d / L2 / L3 read & write bandwidth)
     A grid of subplots, one subplot per fixed block count
     (e.g. ``blocks=1, blocks=2, ...`` or ``blocks=228, blocks=456, ...``).
     Per subplot:  x = repetitions (log2), y = bandwidth (GiB/s),
     one line per thread count.  A single shared legend titled "Threads".

  2. LDS BEST-PER-CONFIGURATION  (LDS / shared memory read & write bandwidth)
     A single panel.  Each line is the *best-thread* run of one
     (array size, allocation type) configuration, e.g. ``16_dyn (T=1024)``.

  3. THREADS PANEL  (NVIDIA L1 read & write bandwidth)
     A single panel with one line per thread count (x = repetitions, log2).
     NVIDIA L1 is measured per-SM (a single block) so its sweep is a 2D
     threads x reps grid, not a block sweep -- it gets its OWN figure and is
     never merged into the LDS figures (matches thesis Fig. 4.9 / 4.10).

CLASSIFICATION is by benchmark type (filename token), not by grid dimensionality:
``L2``/``L3``/``vL1d``/``sL1d`` -> block-sweep grid; ``L1`` -> threads panel;
``LDS`` -> best-per-configuration.  Grid dimensionality is only a fallback for
unrecognised tokens.

DATA SOURCE
-----------
The script consumes the exact CSV grids produced by
``util::writeBandwidthGridToCSV`` (see ``src/utils/printing.hpp``), i.e. the
same raw sweep measurements used to pick the optimal thread/block counts and the
peak bandwidth -- curves are NEVER reconstructed from summary statistics.

  * 3D long format (block sweep), header ``blocks,threads,reps,bandwidth``:
        blocks   -> subplot
        threads  -> line (legend "Threads")
        reps     -> x-axis (log2, GiB/s on y)
        bandwidth-> y-axis value [GiB/s]

  * 2D wide format (single config / LDS), header ``threads,<rep>,<rep>,...``:
        first column 'threads' -> one row per thread count
        remaining columns      -> repetition counts (the header values)
        cell                   -> bandwidth [GiB/s]
     The "best" thread row (max bandwidth over all reps) becomes one LDS line.

USAGE
-----
    # one block-sweep figure from a single 3D grid CSV
    plot_bandwidth.py blocksweep --input GPU__L2_Read_BW_Grid.csv \
        --title "MI300A: L2 read bandwidth (block sweep)" --outdir out/

    # one LDS best-per-config figure from several 2D grid CSVs
    plot_bandwidth.py lds --direction read --gpu MI300A \
        --input GPU__LDS_Read_16KiB_dyn_BW_Grid.csv \
        --input GPU__LDS_Read_32KiB_stat_BW_Grid.csv --outdir out/

    # auto: scan a results directory and regenerate every bandwidth figure
    plot_bandwidth.py auto --indir results/MI300A --outdir results/MI300A

A PNG is written for every figure. Existing files are overwritten by default.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
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
        "axes.titlesize": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,  # grid behind the data lines
        "lines.linewidth": 1.6,
    }
)

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _pow2_label(rep: float) -> str:
    """Format a repetition count as a 2^n superscript label, e.g. 256 -> '2⁸'."""
    exp = int(round(math.log2(rep)))
    return "2" + str(exp).translate(_SUPERSCRIPT)


def _apply_pow2_xaxis(ax, reps: list[float]) -> None:
    """Configure a log2 x-axis with thinned 2^n superscript tick labels.

    Repetitions sweep across many powers of two (2^1 .. 2^18); a log2 scale gives
    them equal spacing exactly like the thesis figures.  We label a thinned
    subset (~8 ticks) to avoid crowding.
    """
    reps = sorted({r for r in reps if r > 0})
    if not reps:
        return
    ax.set_xscale("log", base=2)
    exps = [int(round(math.log2(r))) for r in reps]
    step = max(1, round(len(exps) / 8))
    # Anchor on the largest power and thin downward so the max is always labelled
    # and the final two ticks never crowd (mirrors the thesis 2^2,2^4,...,2^18).
    chosen = sorted({e for e in exps[::-1][::step]})
    ax.set_xticks([2.0**e for e in chosen])
    ax.set_xticklabels(["2" + str(e).translate(_SUPERSCRIPT) for e in chosen])
    ax.xaxis.set_minor_locator(mticker.NullLocator())


# --------------------------------------------------------------------------- #
# Strongly typed in-memory representation of a parsed grid CSV.                 #
# --------------------------------------------------------------------------- #
@dataclass
class BlockSweepGrid:
    """A 3D sweep: bandwidth[block][thread][rep] in GiB/s."""

    blocks: list[int]
    threads: list[int]
    reps: list[int]
    # values[b][t][r] indexed the same way as blocks/threads/reps
    values: list[list[list[float]]]


@dataclass
class ConfigGrid:
    """A 2D sweep for a single configuration: bandwidth[thread][rep] in GiB/s."""

    threads: list[int]
    reps: list[int]
    values: list[list[float]]  # values[t][r]
    size_kib: Optional[int] = None       # array size in KiB (from filename)
    alloc: Optional[str] = None          # 'dyn' | 'stat' (from filename)
    direction: str = "read"              # 'read' | 'write'
    source: str = ""                     # originating file (for diagnostics)

    def best_thread_curve(self) -> tuple[int, list[float]]:
        """Return (best_thread, bandwidth_over_reps) for the peak-bandwidth row.

        Mirrors the C++ optimal search: the configuration's representative curve
        is the thread count that achieved the single highest bandwidth sample.
        """
        best_t_idx, best_val = 0, -1.0
        for ti, row in enumerate(self.values):
            row_max = max(row) if row else -1.0
            if row_max > best_val:
                best_val, best_t_idx = row_max, ti
        return self.threads[best_t_idx], self.values[best_t_idx]

    def label(self) -> str:
        """Legend label, e.g. '16_dyn (T=1024)'."""
        best_t, _ = self.best_thread_curve()
        size = f"{self.size_kib}" if self.size_kib is not None else "?"
        alloc = self.alloc if self.alloc is not None else "?"
        return f"{size}_{alloc} (T={best_t})"

    def sort_key(self) -> tuple:
        """Order configs as 16_dyn, 16_stat, 32_dyn, 32_stat, ... (dyn before stat)."""
        return (self.size_kib or 0, 0 if self.alloc == "dyn" else 1)


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
    """Parse a wide-format ``threads,<rep>,<rep>,...`` CSV (single configuration)."""
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
# Figure 1: block-sweep grid                                                   #
# --------------------------------------------------------------------------- #
def _grid_shape(n: int) -> tuple[int, int]:
    """Choose a (rows, cols) layout: 1xN for small N, else <=3 columns."""
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    cols = 3
    rows = math.ceil(n / cols)
    return rows, cols


def plot_block_sweep(grid: BlockSweepGrid, title: str, outdir: Path, outbase: str, dpi: int) -> None:
    nrows, ncols = _grid_shape(len(grid.blocks))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.7 * nrows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    handles, labels = [], []
    for panel, ax in enumerate(flat):
        if panel >= len(grid.blocks):
            ax.set_visible(False)
            continue
        ax.set_title(f"blocks={grid.blocks[panel]}")
        for ti, thr in enumerate(grid.threads):
            (line,) = ax.plot(grid.reps, grid.values[panel][ti], label=str(thr))
            if panel == 0:
                handles.append(line)
                labels.append(str(thr))
        _apply_pow2_xaxis(ax, grid.reps)

    # Set y-limits explicitly: bottom=0, top=data max + 15% headroom.
    # set_ylim(bottom=0) alone eliminates matplotlib's auto top-margin, so the
    # peak data points end up flush against the top edge and get clipped.
    all_vals = [v for b_vals in grid.values for t_vals in b_vals for v in t_vals
                if not math.isnan(v)]
    ymax = max(all_vals) if all_vals else 1.0
    # Call set_ylim on every axis, not just flat[0]. With sharey=True, calling
    # set_ylim on one axis disables autoscaling only on that axis; the others
    # still fire autoscale during draw() and override the shared viewLim.
    for ax in flat:
        ax.set_ylim(0, ymax * 1.15)

    # Reserve a fixed ~1.1 inch band at the bottom for the shared x-label and
    # legend so they never overlap, regardless of the number of subplot rows.
    fig_h = fig.get_size_inches()[1]
    bottom = 1.05 / fig_h
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=bottom, hspace=0.30, wspace=0.12)

    # Shared decorations -- avoid repeating axis labels on every subplot.
    fig.suptitle(title if title else "Bandwidth")
    fig.supylabel("Bandwidth (GiB/s)")
    fig.supxlabel("Repetitions", y=bottom * 0.55)

    # Single shared legend, horizontal, titled "Threads", below the x-label.
    fig.legend(
        handles,
        labels,
        title="Threads",
        loc="lower center",
        ncol=max(1, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    _save(fig, outdir, outbase, dpi)


# --------------------------------------------------------------------------- #
# Figure 2: LDS best-per-configuration                                         #
# --------------------------------------------------------------------------- #
def plot_lds_best_per_config(
    configs: list[ConfigGrid], gpu: str, direction: str, outdir: Path, outbase: str, dpi: int
) -> None:
    configs = sorted(configs, key=ConfigGrid.sort_key)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    all_reps: list[int] = []
    for cfg in configs:
        _best_t, curve = cfg.best_thread_curve()
        ax.plot(cfg.reps, curve, label=cfg.label())
        all_reps = cfg.reps if len(cfg.reps) > len(all_reps) else all_reps

    _apply_pow2_xaxis(ax, all_reps)
    ax.set_xlabel("Repetitions")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_title(f"{gpu} LDS {direction} BW: best per configuration")
    ncol = 2 if len(configs) > 3 else 1
    ax.legend(ncol=ncol, loc="center right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, outdir, outbase, dpi)


# --------------------------------------------------------------------------- #
# Figure 3: threads panel (NVIDIA L1)                                          #
# --------------------------------------------------------------------------- #
def plot_threads_panel(grid: ConfigGrid, title: str, outdir: Path, outbase: str, dpi: int) -> None:
    """Single panel, one bandwidth-vs-reps line per thread count.

    Used for NVIDIA L1, which is measured per-SM (a single block) and therefore
    produces a 2D threads x reps grid rather than a block sweep. This is a stand-
    alone figure -- NOT folded into the LDS figures -- and mirrors the thesis
    A100 L1 plots (Fig. 4.9 read / 4.10 write): x = repetitions (log2), y =
    bandwidth (GiB/s), legend titled "Threads".  Axis mapping:
        grid.threads[t] -> one line   grid.reps[r] -> x   grid.values[t][r] -> y
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for ti, thr in enumerate(grid.threads):
        ax.plot(grid.reps, grid.values[ti], label=str(thr))
    _apply_pow2_xaxis(ax, grid.reps)
    ax.set_xlabel("Repetitions")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_title(title)
    ncol = 2 if len(grid.threads) > 3 else 1
    ax.legend(title="Threads", ncol=ncol, loc="center right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, outdir, outbase, dpi)


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


# Block-sweep benchmarks (3D grids). vL1d/sL1d are AMD; L2/L3 are both vendors.
_BLOCK_SWEEP_TOKENS = {"vL1d", "sL1d", "L2", "L3"}


def _benchmark_category(name: str) -> tuple[str, str]:
    """Classify a grid file by *benchmark type* (filename token), not dimensionality.

    Returns ``(category, pretty_token)`` where category is one of:
        "block_sweep" -> L2 / L3 / vL1d / sL1d  (3D block-sweep grid figure)
        "l1"          -> NVIDIA L1               (dedicated 2D threads-panel figure)
        "lds"         -> LDS / shared memory     (best-per-configuration figure)
        "unknown"     -> fall back to grid dimensionality
    This is what keeps NVIDIA L1 (also a 2D grid) out of the LDS figures.
    """
    token = _benchmark_token(name)
    if token in _BLOCK_SWEEP_TOKENS:
        return "block_sweep", token
    if token == "L1":
        return "l1", token
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

        # Route by benchmark type. Unknown tokens fall back to dimensionality so
        # arbitrary 3D grids still plot as block sweeps and 2D grids as LDS.
        if category == "block_sweep" or (category == "unknown" and _is_block_sweep(header)):
            grid = load_block_sweep(csv)
            title = f"{gpu}: {token} {direction} bandwidth (block sweep)"
            plot_block_sweep(grid, title, outdir, _slug(f"{gpu} - {token} {direction} bandwidth"), dpi)
            n_figs += 1
        elif category == "l1":
            # NVIDIA L1: dedicated threads-panel figure, never merged into LDS.
            grid = load_config_grid(csv)
            title = f"{gpu}: L1 {direction} bandwidth"
            plot_threads_panel(grid, title, outdir, _slug(f"{gpu} - L1 {direction} bandwidth"), dpi)
            n_figs += 1
        else:  # "lds" or unrecognised 2D grid
            cfg = load_config_grid(csv)
            lds_configs[cfg.direction].append(cfg)

    # Combine the LDS/shared configs into one best-per-configuration figure/direction.
    for direction, configs in lds_configs.items():
        if not configs:
            continue
        gpu = _gpu_from_name(Path(configs[0].source).name)
        plot_lds_best_per_config(configs, gpu, direction, outdir, _slug(f"{gpu} - LDS {direction} bandwidth"), dpi)
        n_figs += 1

    print(f"[plot_bandwidth] generated {n_figs} figure(s) into {outdir}")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    bs = sub.add_parser("blocksweep", help="one block-sweep grid from a 3D grid CSV")
    bs.add_argument("--input", required=True, help="blocks,threads,reps,bandwidth CSV")
    bs.add_argument("--title", default="Bandwidth")
    bs.add_argument("--outdir", default=".")
    bs.add_argument("--outfile", default=None, help="output file stem (no extension)")
    bs.add_argument("--dpi", type=int, default=150)

    ld = sub.add_parser("lds", help="LDS best-per-config figure from one or more 2D grid CSVs")
    ld.add_argument("--input", action="append", required=True, help="threads,<rep>,... CSV (repeatable)")
    ld.add_argument("--gpu", default="GPU")
    ld.add_argument("--direction", default=None, choices=["read", "write"])
    ld.add_argument("--outdir", default=".")
    ld.add_argument("--outfile", default=None)
    ld.add_argument("--dpi", type=int, default=150)

    l1 = sub.add_parser("l1", help="dedicated L1 threads-panel figure from one 2D grid CSV")
    l1.add_argument("--input", required=True, help="threads,<rep>,... CSV")
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
        plot_block_sweep(grid, args.title, Path(args.outdir), outbase, args.dpi)
        return 0

    if args.mode == "lds":
        configs = [load_config_grid(Path(p)) for p in args.input]
        direction = args.direction or (configs[0].direction if configs else "read")
        outbase = args.outfile or _slug(f"{args.gpu} - LDS {direction} bandwidth")
        plot_lds_best_per_config(configs, args.gpu, direction, Path(args.outdir), outbase, args.dpi)
        return 0

    if args.mode == "l1":
        path = Path(args.input)
        grid = load_config_grid(path)
        gpu = _gpu_from_name(path.name)
        title = args.title or f"{gpu}: L1 {grid.direction} bandwidth"
        outbase = args.outfile or _slug(f"{gpu} - L1 {grid.direction} bandwidth")
        plot_threads_panel(grid, title, Path(args.outdir), outbase, args.dpi)
        return 0

    if args.mode == "auto":
        indir = Path(args.indir)
        outdir = Path(args.outdir) if args.outdir else indir
        return run_auto(indir, outdir, args.dpi)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
