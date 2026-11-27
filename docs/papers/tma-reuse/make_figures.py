"""Draw the paper's figures from the H100 tune results.

Reads the sweep .csv files in cuda/tests/results/H100/ and writes fig_levers.pdf
(swizzle and depth gains, by kernel) and fig_products.pdf (the same, by matrix
product), each also as .svg for the web, next to this script. Also prints the numbers the paper's tables quote.

    python docs/papers/tma-reuse/make_figures.py
"""

import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "cuda" / "tests" / "results" / "H100"

SWIZZLE_COLOR = "#2a78d6"  # categorical slot 1
DEPTH_COLOR = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def best_by(rows, key_of, ms_of):
    """Smallest time in rows for each key."""
    best_dict = collections.defaultdict(lambda: float("inf"))
    for row in rows:
        key_val = key_of(row)
        if key_val is not None:
            best_dict[key_val] = min(best_dict[key_val], ms_of(row))
    return best_dict


def matmul_gains():
    """Per product: (swizzle gain, depth gain, best swizzled row)."""
    path = RESULTS / "mma_tma_matmul_bf16_model_2026-11-23_fixed.csv"
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ran" and r["same"] == "1"]
    ms = lambda r: float(r["first_ms"])  # noqa: E731
    swizzle_best = best_by(rows, lambda r: (r["product"], r["swizzle"]), ms)
    depth_best = best_by(rows, lambda r: (r["product"], r["parts"] == "1"), ms)
    product_names = sorted({r["product"] for r in rows})
    gains_dict = {}
    for name in product_names:
        best_row = min((r for r in rows if r["product"] == name and r["swizzle"] == "1"), key=ms)
        gains_dict[name] = (
            swizzle_best[(name, "0")] / swizzle_best[(name, "1")],
            depth_best[(name, True)] / depth_best[(name, False)],
            best_row,
        )
    total_swizzled = sum(swizzle_best[(n, "1")] for n in product_names)
    total_plain = sum(swizzle_best[(n, "0")] for n in product_names)
    total_one_part = sum(depth_best[(n, True)] for n in product_names)
    total_more_parts = sum(depth_best[(n, False)] for n in product_names)
    totals = (total_plain / total_swizzled, total_one_part / total_more_parts)
    return gains_dict, totals, (total_swizzled, total_plain, total_one_part, total_more_parts)


def encoder_gains(file_name):
    """(swizzle gain, depth gain) of one embedding-lookup sweep."""
    rows = [r for r in csv.DictReader(open(RESULTS / file_name)) if r["same"] == "1" and float(r["first_ms"]) > 0]
    ms = lambda r: float(r["first_ms"])  # noqa: E731
    swizzle_best = best_by(rows, lambda r: r["swizzle"], ms)
    depth_best = best_by(rows, lambda r: r["stages"] == "1", ms)
    return swizzle_best["0"] / swizzle_best["128"], depth_best[True] / depth_best[False], swizzle_best, depth_best


def adamw_gains():
    """(depth gain, TMA-store gain) of the AdamW sweep."""
    rows = [r for r in csv.DictReader(open(RESULTS / "adamw_tma_bf16_2026-11-22_144452.csv")) if r["status"] == "ran"]
    ms = lambda r: float(r["first_ms"])  # noqa: E731
    depth_best = best_by(rows, lambda r: r["parts"] == "1" if r["store"] == "tma" else None, ms)
    store_best = best_by(rows, lambda r: r["store"], ms)
    return depth_best[True] / depth_best[False], store_best["threads"] / store_best["tma"], depth_best, store_best


def dot_rows(ax, labels, swizzle_list, depth_list):
    """Two markers a row on a log axis, with a reference line at 1."""
    y_list = list(range(len(labels)))[::-1]
    ax.axvline(1.0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
    for y_val, s_val, d_val in zip(y_list, swizzle_list, depth_list):
        ax.axhline(y_val, color=GRID, linewidth=0.6, zorder=0)
        if s_val is not None:
            ax.plot(s_val, y_val, "o", color=SWIZZLE_COLOR, markersize=5, markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.plot(d_val, y_val, "D", color=DEPTH_COLOR, markersize=4.5, markeredgecolor="white", markeredgewidth=1, zorder=3)
    ax.set_yticks(y_list)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ticks_list = [1.0, 1.1, 1.25, 1.5, 1.75]
    ax.set_xticks(ticks_list)
    ax.set_xticklabels([f"{t:g}×" for t in ticks_list])
    ax.minorticks_off()
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], "o", color=SWIZZLE_COLOR, label="128-byte swizzle: best without / best with")
    ax.plot([], [], "D", color=DEPTH_COLOR, label="depth: best at 1 stage / best at 2 or more")
    ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)


def main():
    gains_dict, totals, total_ms = matmul_gains()
    enc_list = [
        ("embedding, width 1,536", "encoder_tma_bf16_2026-11-21_055718.csv"),
        ("embedding, width 6,144", "encoder_tma_bf16_2026-11-21_060305.csv"),
        ("embedding, width 15,360", "encoder_tma_bf16_2026-11-21_061414.csv"),
    ]
    adamw_depth, adamw_store, adamw_depth_ms, adamw_store_ms = adamw_gains()

    print("matmul totals ms (swizzled, unswizzled, 1 part, 2+ parts):", [round(t, 2) for t in total_ms])
    print("matmul gains (swizzle, depth):", [round(t, 3) for t in totals])
    for name, (s_val, d_val, row) in sorted(gains_dict.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:30s} swizzle {s_val:.2f} depth {d_val:.2f}  tile {row['block_rows']}x{row['block_columns']}"
              f" warps {row['warps_down']}x{row['warps_across']} parts {row['parts']}")
    enc_gains_list = []
    for label, file_name in enc_list:
        s_val, d_val, s_ms, d_ms = encoder_gains(file_name)
        enc_gains_list.append((label, s_val, d_val))
        print(f"{label}: swizzle {s_val:.3f} depth {d_val:.3f}  ms {dict(s_ms)} {dict(d_ms)}")
    print(f"adamw: depth {adamw_depth:.3f} store {adamw_store:.3f}  ms {dict(adamw_depth_ms)} {dict(adamw_store_ms)}")

    # Figure 1: the two levers, kernel by kernel.
    labels = ["AdamW step (GPT-2 small)"] + [g[0] for g in enc_gains_list] + ["matrix products (all 15)"]
    swizzle_list = [None] + [g[1] for g in enc_gains_list] + [totals[0]]
    depth_list = [adamw_depth] + [g[2] for g in enc_gains_list] + [totals[1]]
    fig, ax = plt.subplots(figsize=(5.5, 2.1))
    dot_rows(ax, labels, swizzle_list, depth_list)
    ax.set_xlim(0.93, 1.8)
    ax.set_xlabel("speedup from the lever (best setting with it against best without)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_levers.pdf")
    fig.savefig(HERE / "fig_levers.svg")

    # Figure 2: the two levers, product by product.
    ordered = sorted(gains_dict.items(), key=lambda kv: -kv[1][0])
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    dot_rows(ax, [kv[0] for kv in ordered], [kv[1][0] for kv in ordered], [kv[1][1] for kv in ordered])
    ax.set_xlim(0.97, 1.85)
    ax.set_xlabel("speedup from the lever (best setting with it against best without)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_products.pdf")
    fig.savefig(HERE / "fig_products.svg")


if __name__ == "__main__":
    main()
