"""Summary table, LaTeX tables and figures of the report.

Usage (from any directory):  python analysis/plot_results.py [results_dir]

Reads   raw_measurements.csv, process_memory.csv, run_status.csv
Writes  summary.csv, plot_*.png and tables/*.tex (included by the report)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, LogLocator, NullLocator

RESULTS_DIR = (sys.argv[1] if len(sys.argv) > 1 else
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

COLORS = {"ijk": "#5baa5b", "ikj": "#4c8eda", "blocked": "#c0504d", "transposed": "#e07b39",
          "csr": "#7b5ea7", "convert": "#999999"}
LABELS = {"ijk": "ijk (baseline)", "ikj": "ikj (loop reordering)",
          "blocked": "blocked ikj (cache blocking)", "transposed": "ijk on $B^T$ (transposed)"}
DENSE_ORDER = ["ijk", "ikj", "blocked", "transposed"]
BUILD_STYLE = {"scalar": "-", "avx2": "--"}
DIST_LABELS = {"uniform": "Uniform", "band": "Band"}
MIB = 2 ** 20

KEYS = ["experiment", "method", "build", "n", "distribution", "density", "block"]


def path(*parts):
    return os.path.join(RESULTS_DIR, *parts)


def add_grid(ax, xs=None, xlabels=None):
    """One labelled tick per measured x (if given), one horizontal line per decade."""
    if xs is not None:
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FixedFormatter(xlabels or [f"{x:g}" for x in xs]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    lo, hi = ax.get_ylim()
    if np.log10(hi / lo) < 4:   # few decades: add 2 and 5 marks so values are easy to read
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=30))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        nice = [m * 10.0 ** e for e in range(-8, 9) for m in (1, 2, 5)]
        ax.set_ylim(bottom=max(v for v in nice if v <= lo))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)


# --------------------------------------------------------------------------- data

def load_raw():
    raw = pd.read_csv(path("raw_measurements.csv"), float_precision="round_trip",
                      dtype={"density": str})
    raw["block"] = raw["block"].fillna(0).astype(int)
    return raw


def load_peaks():
    peaks = pd.read_csv(path("process_memory.csv"), dtype={"density": str})
    peaks["block"] = peaks["block"].fillna(0).astype(int)
    return peaks


def load_status():
    status = pd.read_csv(path("run_status.csv"), dtype={"density": str})
    status["block"] = status["block"].fillna(0).astype(int)
    return status


def summarise(raw, peaks):
    """One row per configuration. Time statistics use the measured runs of all processes;
    warm-up and check runs are excluded."""
    measured = raw[raw["run_type"] == "measured"]
    summary = (
        measured.groupby(KEYS)
        .agg(runs=("time_seconds", "size"),
             processes=("process", "nunique"),
             batch_size=("batch_size", "max"),
             median_time=("time_seconds", "median"),
             q1_time=("time_seconds", lambda x: x.quantile(0.25)),
             q3_time=("time_seconds", lambda x: x.quantile(0.75)),
             nnz_a=("nnz_a", "first"), nnz_b=("nnz_b", "first"), nnz_c=("nnz_c", "first"),
             storage_bytes=("memory_bytes", "median"))
        .reset_index()
    )
    per_process = measured.groupby(KEYS + ["process"])["time_seconds"].median()
    spread = per_process.groupby(KEYS).agg(lambda x: x.max() - x.min())
    summary = summary.merge(spread.rename("process_spread").reset_index(), on=KEYS)
    peak = peaks.groupby(KEYS)["peak_bytes"].median().rename("peak_bytes")
    summary = summary.merge(peak.reset_index(), on=KEYS, how="left")

    summary["density_value"] = summary["density"].astype(float)
    summary["iqr_time"] = summary["q3_time"] - summary["q1_time"]
    summary["iqr_over_median_pct"] = 100 * summary["iqr_time"] / summary["median_time"]
    summary["process_spread_pct"] = 100 * summary["process_spread"] / summary["median_time"]
    summary["ns_per_multiply_add"] = summary["median_time"] * 1e9 / summary["n"] ** 3
    summary["density_a_pct"] = 100 * summary["nnz_a"] / summary["n"] ** 2
    summary["density_c_pct"] = 100 * summary["nnz_c"] / summary["n"] ** 2
    base = summary[(summary["experiment"] == "dense") & (summary["method"] == "ijk")]
    base = base.set_index("n")["median_time"]
    dense = summary["experiment"] == "dense"
    summary.loc[dense, "speedup_vs_ijk"] = summary.loc[dense, "n"].map(base) \
        / summary.loc[dense, "median_time"]
    return summary


def select(summary, **kw):
    d = summary
    for k, v in kw.items():
        d = d[d[k] == v]
    return d


def one(summary, column, **kw):
    d = select(summary, **kw)
    return None if d.empty else d.iloc[0][column]


# --------------------------------------------------------------------------- figures

def save(fig, name):
    fig.tight_layout()
    fig.savefig(path(name), dpi=150)
    plt.close(fig)
    print(f"Saved {path(name)}")


def errorbar(ax, d, x, **style):
    ax.errorbar(d[x], d["median_time"],
                yerr=[d["median_time"] - d["q1_time"], d["q3_time"] - d["median_time"]],
                capsize=3, marker="o", markersize=4, linewidth=1.5, **style)


def plot_dense_time(summary):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    d0 = select(summary, experiment="dense", build="scalar")
    for m in DENSE_ORDER:
        d = d0[d0["method"] == m].sort_values("n")
        errorbar(ax, d, "n", color=COLORS[m], label=LABELS[m])
    ns = np.sort(d0["n"].unique())
    n0 = 500 if 500 in ns else ns[0]
    anchor = one(summary, "median_time", experiment="dense", method="ijk", n=n0)
    ax.plot(ns, anchor * (ns / n0) ** 3, color="gray", linewidth=0.8, linestyle=":",
            label=f"$n^3$ through ijk at n = {n0}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time per multiplication (s, log scale)", fontsize=12)
    ax.set_title("Dense methods, scalar build", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, ns, [str(int(n)) for n in ns])
    save(fig, "plot_dense_time.png")


def plot_cost_per_op(summary):
    """Time / n^3 in nanoseconds per multiply-add, both builds. Flat = cubic scaling."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    d0 = select(summary, experiment="dense")
    d0 = d0[d0["method"] != "transpose"]
    for build in ["scalar", "avx2"]:
        for m in DENSE_ORDER:
            d = d0[(d0["method"] == m) & (d0["build"] == build)].sort_values("n")
            if d.empty:
                continue
            ax.plot(d["n"], d["ns_per_multiply_add"], color=COLORS[m],
                    linestyle=BUILD_STYLE[build], marker="o" if build == "scalar" else "s",
                    markersize=4, linewidth=1.5, label=f"{m} ({build})")
    ns = np.sort(d0["n"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time / $n^3$ (ns per multiply-add, log scale)", fontsize=12)
    ax.set_title("Cost per operation (solid: scalar build, dashed: AVX2 build)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    add_grid(ax, ns, [str(int(n)) for n in ns])
    save(fig, "plot_cost_per_op.png")


def plot_block_size(summary):
    fig, ax = plt.subplots(figsize=(7, 4.0))
    d0 = select(summary, experiment="blocksize")
    sizes = sorted(d0["n"].unique())
    colors = dict(zip(sizes, ["#4c8eda", "#c0504d", "#5baa5b"]))
    for build in ["scalar", "avx2"]:
        for n in sizes:
            d = d0[(d0["build"] == build) & (d0["n"] == n)].sort_values("block")
            errorbar(ax, d.assign(**{"x": d["block"]}), "x", color=colors[n],
                     linestyle=BUILD_STYLE[build], label=f"n = {n} ({build})")
    bs = np.sort(d0["block"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block size b (log scale)", fontsize=12)
    ax.set_ylabel("Median time (s, log scale)", fontsize=12)
    ax.set_title("Blocked kernel: effect of the block size", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    add_grid(ax, bs, [str(int(b)) for b in bs])
    save(fig, "plot_block_size.png")


def density_ticks(ds):
    return [f"{100 * d:g}" for d in ds]


def plot_density_time(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    d0 = select(summary, experiment="density")
    n = int(d0["n"].iloc[0])
    for ax, dist in zip(axes, ["uniform", "band"]):
        dd = d0[d0["distribution"] == dist]
        csr = dd[dd["method"] == "csr"].sort_values("density_value")
        conv = dd[dd["method"] == "convert"].set_index("density")["median_time"]
        errorbar(ax, csr, "density_value", color=COLORS["csr"], label="CSR kernel")
        ax.plot(csr["density_value"], csr["median_time"] + csr["density"].map(conv).values,
                color=COLORS["csr"], linestyle=":", marker="^", markersize=4,
                label="CSR kernel + conversion of A and B")
        for build in ["scalar", "avx2"]:
            b = dd[(dd["method"] == "blocked") & (dd["build"] == build)].sort_values("density_value")
            errorbar(ax, b, "density_value", color=COLORS["blocked"], linestyle=BUILD_STYLE[build],
                     label=f"dense blocked ({build})")
        ijk = dd[dd["method"] == "ijk"].sort_values("density_value")
        ax.plot(ijk["density_value"], ijk["median_time"], color=COLORS["ijk"], marker="D",
                linestyle="none", markersize=5, label="dense ijk (scalar)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Density of A and B (% nonzero, log scale)", fontsize=11)
        ax.set_title(f"{DIST_LABELS[dist]} nonzeros, n = {n}", fontsize=12)
        ds = np.sort(csr["density_value"].unique())
        add_grid(ax, ds, density_ticks(ds))
    axes[0].set_ylabel("Median time (s, log scale)", fontsize=11)
    axes[0].legend(fontsize=8)
    save(fig, "plot_density_time.png")


def plot_output_density(summary):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    d0 = select(summary, experiment="density", method="csr")
    n = int(d0["n"].iloc[0])
    ds = np.sort(d0["density_value"].unique())
    model = np.geomspace(ds.min(), ds.max(), 200)
    ax.plot(100 * model, 100 * (1 - (1 - model ** 2) ** n), color="gray", linestyle=":",
            linewidth=1, label=f"uniform model $1-(1-d^2)^n$")
    ax.plot(100 * model, 100 * model, color="gray", linestyle="--", linewidth=0.8,
            label="output density = input density")
    for dist, color in [("uniform", "#7b5ea7"), ("band", "#e07b39")]:
        d = d0[d0["distribution"] == dist].sort_values("density_value")
        ax.plot(d["density_a_pct"], d["density_c_pct"], marker="o", color=color, linewidth=1.5,
                label=f"{DIST_LABELS[dist]}: measured nnz(C) / $n^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density of A (% nonzero, log scale)", fontsize=12)
    ax.set_ylabel("Density of C = AB (% nonzero, log scale)", fontsize=12)
    ax.set_title(f"Fill-in of the product, n = {n}", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, 100 * ds, [f"{100 * d:g}" for d in ds])
    save(fig, "plot_output_density.png")


def plot_density_memory(summary):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    d0 = select(summary, experiment="density")
    n = int(d0["n"].iloc[0])
    for dist, color in [("uniform", "#7b5ea7"), ("band", "#e07b39")]:
        d = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")].sort_values("density_value")
        ax.plot(d["density_value"], d["storage_bytes"] / MIB, color=color, marker="o",
                linewidth=1.5, label=f"CSR {dist}: computed storage (A, B, C, work space)")
        ax.plot(d["density_value"], d["peak_bytes"] / MIB, color=color, marker="s",
                linestyle=":", linewidth=1.2, label=f"CSR {dist}: peak working set")
    b = d0[(d0["method"] == "blocked") & (d0["build"] == "scalar")
           & (d0["distribution"] == "uniform")].sort_values("density_value")
    ax.plot(b["density_value"], b["peak_bytes"] / MIB, color=COLORS["blocked"], marker="s",
            linestyle=":", linewidth=1.2, label="dense blocked: peak working set")
    ax.axhline(24 * n ** 2 / MIB, color="gray", linestyle="--", linewidth=1,
               label="24$n^2$ bytes (three float64 matrices)")
    ds = np.sort(d0["density_value"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density of A and B (% nonzero, log scale)", fontsize=12)
    ax.set_ylabel("Memory (MiB, log scale)", fontsize=12)
    ax.set_title(f"Memory of the sparse and dense methods, n = {n}", fontsize=12)
    ax.legend(fontsize=8)
    add_grid(ax, ds, density_ticks(ds))
    save(fig, "plot_density_memory.png")


def plot_size_time(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d0 = select(summary, experiment="size")
    styles = {("uniform", "0.001"): ("#7b5ea7", "-"), ("uniform", "0.01"): ("#7b5ea7", "--"),
              ("band", "0.001"): ("#e07b39", "-"), ("band", "0.01"): ("#e07b39", "--")}
    for (dist, dens), (color, ls) in styles.items():
        d = d0[(d0["method"] == "csr") & (d0["distribution"] == dist) & (d0["density"] == dens)]
        errorbar(ax, d.sort_values("n"), "n", color=color, linestyle=ls,
                 label=f"CSR, {dist}, {100 * float(dens):g} %")
    b = d0[(d0["method"] == "blocked") & (d0["distribution"] == "uniform")
           & (d0["density"] == "0.01")].sort_values("n")
    errorbar(ax, b, "n", color=COLORS["blocked"], label="dense blocked (scalar), uniform 1 %")
    ns = np.sort(d0["n"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time per multiplication (s, log scale)", fontsize=12)
    ax.set_title("Sparse multiplication at fixed density", fontsize=12)
    ax.legend(fontsize=8)
    add_grid(ax, ns, [str(int(n)) for n in ns])
    save(fig, "plot_size_time.png")


# --------------------------------------------------------------------------- LaTeX tables

def fmt_time(t):
    if t is None or pd.isna(t):
        return "---"
    if t >= 100:
        return f"{t:.0f}"
    if t >= 10:
        return f"{t:.1f}"
    if t >= 0.01:
        return f"{t:#.3g}"
    mantissa, exponent = f"{t:.2e}".split("e")
    return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$"


def fmt_sig(x, digits=3):
    return "---" if x is None or pd.isna(x) else f"{float(f'{x:.{digits}g}'):g}"


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "---"
    return f"{x:.2g}" if x < 1 else f"{x:.3g}" if x < 99.95 else "100"


def write_table(name, lines):
    os.makedirs(path("tables"), exist_ok=True)
    with open(path("tables", name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path('tables', name)}")


def time_cell(summary, status, **kw):
    t = one(summary, "median_time", **kw)
    if t is None:
        if status is None:
            return "---"
        st = select(status, **kw)
        st = st[st["process"] == 1]
        return st["status"].iloc[0].replace("_", "\\_") if not st.empty else "---"
    return f"{fmt_time(t)} ({one(summary, 'iqr_over_median_pct', **kw):.1f}\\%)"


def table_dense_runtime(summary, status):
    """Scalar build: median (IQR) of the four kernels and the transposition alone."""
    lines = []
    for n in sorted(select(status, experiment="dense")["n"].unique()):
        cells = [time_cell(summary, status, experiment="dense", build="scalar", method=m, n=n)
                 for m in DENSE_ORDER + ["transpose"]]
        lines.append(f"{n} & " + " & ".join(cells) + " \\\\")
    write_table("dense_runtime.tex", lines)


def table_dense_speedup(summary):
    """ns per multiply-add and speedup over ijk, both builds."""
    lines = []
    cols = [("scalar", "ijk"), ("scalar", "ikj"), ("scalar", "blocked"), ("scalar", "transposed"),
            ("avx2", "ikj"), ("avx2", "blocked"), ("avx2", "transposed")]
    for n in sorted(select(summary, experiment="dense")["n"].unique()):
        ns = [fmt_sig(one(summary, "ns_per_multiply_add", experiment="dense", build=b,
                          method=m, n=n)) for b, m in cols]
        sp = [fmt_sig(one(summary, "speedup_vs_ijk", experiment="dense", build=b, method=m,
                          n=n), 2) for b, m in cols[1:]]
        lines.append(f"{n} & " + " & ".join(ns) + " & " + " & ".join(sp) + " \\\\")
    write_table("dense_speedup.tex", lines)


def table_block_size(summary):
    lines = []
    d0 = select(summary, experiment="blocksize")
    for b in sorted(d0["block"].unique()):
        cells = [time_cell(summary, None, experiment="blocksize", build=build, n=n, block=b)
                 for build in ["scalar", "avx2"] for n in sorted(d0["n"].unique())]
        lines.append(f"{b} & " + " & ".join(cells) + " \\\\")
    write_table("block_size.tex", lines)


def table_density(summary):
    """Per distribution and density: nnz, output density, times and ratios."""
    lines = []
    d0 = select(summary, experiment="density")
    for dist in ["uniform", "band"]:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{DIST_LABELS[dist]} nonzeros}}}} \\\\")
        dd = d0[d0["distribution"] == dist]
        for dens in sorted(dd["density"].unique(), key=float):
            kw = dict(experiment="density", distribution=dist, density=dens)
            csr = one(summary, "median_time", method="csr", **kw)
            conv = one(summary, "median_time", method="convert", **kw)
            bs = one(summary, "median_time", method="blocked", build="scalar", **kw)
            bv = one(summary, "median_time", method="blocked", build="avx2", **kw)
            cells = [f"{100 * float(dens):g}",
                     fmt_pct(one(summary, "density_a_pct", method="csr", **kw)),
                     fmt_pct(one(summary, "density_c_pct", method="csr", **kw)),
                     fmt_time(csr), fmt_time(conv), fmt_time(bs), fmt_time(bv),
                     fmt_sig(bs / csr, 3), fmt_sig(bs / (csr + conv), 3)]
            lines.append(" & ".join(cells) + " \\\\")
        if dist == "uniform":
            lines.append("\\addlinespace")
    write_table("density.tex", lines)


def table_size(summary, status):
    lines = []
    st = select(status, experiment="size", method="csr")
    for dist in ["uniform", "band"]:
        for dens in ["0.001", "0.01"]:
            for n in sorted(st["n"].unique()):
                kw = dict(experiment="size", method="csr", distribution=dist, density=dens, n=n)
                t = one(summary, "median_time", **kw)
                peak = one(summary, "peak_bytes", **kw)
                storage = one(summary, "storage_bytes", **kw)
                b = one(summary, "median_time", experiment="size", method="blocked",
                        distribution=dist, density=dens, n=n)
                cells = [DIST_LABELS[dist] if n == st["n"].min() and dens == "0.001" else "",
                         f"{100 * float(dens):g}" if n == st["n"].min() else "", str(n),
                         time_cell(summary, status, **kw),
                         fmt_pct(one(summary, "density_c_pct", **kw)),
                         "---" if storage is None else f"{storage / MIB:.0f}",
                         "---" if peak is None or pd.isna(peak) else f"{peak / MIB:.0f}",
                         fmt_time(b)]
                lines.append(" & ".join(cells) + " \\\\")
            if not (dist == "band" and dens == "0.01"):
                lines.append("\\addlinespace")
    write_table("size.tex", lines)


def table_memory_dense(summary):
    """Peak working set of the dense methods (MiB) and the computed storage."""
    lines = []
    for n in sorted(select(summary, experiment="dense")["n"].unique()):
        cells = []
        for m in DENSE_ORDER:
            p = one(summary, "peak_bytes", experiment="dense", build="scalar", method=m, n=n)
            cells.append("---" if p is None or pd.isna(p) else f"{p / MIB:.1f}")
        cells.append(f"{24 * n ** 2 / MIB:.1f}")
        lines.append(f"{n} & " + " & ".join(cells) + " \\\\")
    write_table("memory_dense.tex", lines)


if __name__ == "__main__":
    raw = load_raw()
    summary = summarise(raw, load_peaks())
    status = load_status()
    summary.to_csv(path("summary.csv"), index=False, float_format="%.6g")
    print(f"Saved {path('summary.csv')}")
    plot_dense_time(summary)
    plot_cost_per_op(summary)
    plot_block_size(summary)
    plot_density_time(summary)
    plot_output_density(summary)
    plot_density_memory(summary)
    plot_size_time(summary)
    table_dense_runtime(summary, status)
    table_dense_speedup(summary)
    table_block_size(summary)
    table_density(summary)
    table_size(summary, status)
    table_memory_dense(summary)
