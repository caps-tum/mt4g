#pragma once
inline const char* chartScript = R"PY(
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

dpi = 100
width_in = 1600 / dpi
height_in = width_in * 9 / 32  # keep existing aspect

plt.rcParams.update({
    "figure.figsize": (width_in, height_in),
    "figure.dpi": dpi,
    "font.family": "sans-serif",
    "font.size": 20,                   # base font size
    "axes.titlesize": 22,              # title size
    "axes.labelsize": 22,              # axis label size
    "legend.fontsize": 18,             # legend size
    "xtick.labelsize": 18,             # tick labels
    "ytick.labelsize": 18,
    "grid.linewidth": 1
})

parser = argparse.ArgumentParser()
parser.add_argument('--title', default='Benchmark')
parser.add_argument('--xlabel', default='X')
parser.add_argument('--ylabel', default='Value')
parser.add_argument('--redylabel', default='Reduction Value')
parser.add_argument('--highlights', nargs='*', type=float, default=[])
parser.add_argument('--quantile', type=float, default=0.99)
parser.add_argument('--outdir', default='.')
parser.add_argument('--reduction', action='store_true')
args = parser.parse_args()

df = pd.read_csv(sys.stdin)
x = df.iloc[:, 0]

line_width = 3  # thicker lines

if args.reduction:
    data_cols = df.columns[1:4]
    red_col = df.columns[4]
    # compute clipping thresholds
    thresh_left = df[data_cols].stack().quantile(args.quantile)
    thresh_right = df[red_col].quantile(args.quantile)
    df[data_cols] = df[data_cols].clip(upper=thresh_left)
    df[red_col] = df[red_col].clip(upper=thresh_right)

    # main + twin axis with grid
    fig, ax1 = plt.subplots()
    ax1.set_axisbelow(True)            # grid behind data
    ax1.grid(True)                     # enable grid
    for col in data_cols:
        ax1.plot(x, df[col], label=col, linewidth=line_width)
    ax2 = ax1.twinx()
    ax2.plot(x, df[red_col], label=red_col, linestyle='--',
             linewidth=line_width, color='purple')

    ax1.set_ylabel(args.ylabel)
    ax2.set_ylabel(args.redylabel)
    ax1.set_ylim(top=thresh_left * 1.02)
    ax2.set_ylim(top=thresh_right * 1.02)
    ax2.spines['right'].set_color('purple')
    ax2.tick_params(axis='y', colors='purple')
    ax2.yaxis.label.set_color('purple')

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    if len(all_lines) >= 8:
        # split legend into two columns to preserve layout
        ax1.legend(all_lines, all_labels,
                   loc='center left', bbox_to_anchor=(1, 0.5),
                   ncol=2)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        ax1.legend(all_lines, all_labels, loc='upper left')
    ax = ax1

else:
    data_cols = df.columns[1:]
    thresh = df[data_cols].stack().quantile(args.quantile)
    df[data_cols] = df[data_cols].clip(upper=thresh)

    # single axis plot with grid
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(True)
    for col in data_cols:
        ax.plot(x, df[col], label=col, linewidth=line_width)

    if len(data_cols) > 1:
        if len(data_cols) >= 8:
            # two-column legend for many entries
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            fig.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            ax.legend(loc='upper left')

    ax.set_ylabel(args.ylabel)
    ax.set_ylim(top=thresh * 1.02)

# highlight vertical lines
for hx in args.highlights:
    ax.axvline(x=hx, linestyle='--', linewidth=line_width)

ax.set_xlabel(args.xlabel)
ax.set_title(args.title)

plt.tight_layout()
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / f"{args.title.replace("-", "").replace(" ", "_")}.png", dpi=dpi)
)PY";
