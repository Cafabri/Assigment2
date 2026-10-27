"""Summary table, LaTeX tables and figures of the report.

Usage (from any directory):  python analysis/plot_results.py [results_dir]

Reads   raw_measurements.csv, process_memory.csv, run_status.csv of results_dir and, if it
        exists, of results_dir/extension (the second measurement session, run with
        run_experiment.py --extension); the column "session" tells them apart
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
SESSIONS = {"first": "", "extension": "extension"}

COLORS = {"ijk": "#5baa5b", "ikj": "#4c8eda", "blocked": "#c0504d", "transposed": "#e07b39",
          "csr": "#7b5ea7", "convert": "#999999"}
LABELS = {"ijk": "ijk (baseline)", "ikj": "ikj (loop reordering)",
          "blocked": "blocked ikj (cache blocking)", "transposed": "ijk on $B^T$ (transposed)"}
DENSE_ORDER = ["ijk", "ikj", "blocked", "transposed"]
BUILD_STYLE = {"scalar": "-", "sse2": ":", "avx2": "--"}
BUILD_MARKER = {"scalar": "o", "sse2": "^", "avx2": "s"}
BUILD_ORDER = ["scalar", "sse2", "avx2"]
DIST_LABELS = {"uniform": "Uniform", "band": "Band"}
MIB = 2 ** 20

# The figures are printed at 5-6.3 inches in the report; these sizes keep the text legible there
plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10})

KEYS = ["experiment", "method", "build", "n", "distribution", "density", "block"]


def path(*parts):
    return os.path.join(RESULTS_DIR, *parts)


def add_grid(ax, xs=None, xlabels=None):
    """One tick per measured x (if given), labelled unless the label would overlap the
    previous one (log distance below 7 % of the axis span); one horizontal line per decade."""
    if xs is not None:
        labels = list(xlabels or [f"{x:g}" for x in xs])
        logs = np.log10(np.asarray(xs, dtype=float))
        gap = 0.07 * (logs.max() - logs.min())
        last = None
        for i, lx in enumerate(logs):
            if last is not None and lx - last < gap:
                labels[i] = ""
            else:
                last = lx
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
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

def load(name, **kw):
    """One CSV of every session that has it, with a column "session"."""
    frames = [pd.read_csv(path(folder, name), dtype={"density": str}, **kw).assign(session=s)
              for s, folder in SESSIONS.items() if os.path.exists(path(folder, name))]
    d = pd.concat(frames, ignore_index=True)
    d["block"] = d["block"].fillna(0).astype(int)
    return d


def load_raw():
    return load("raw_measurements.csv", float_precision="round_trip")


def load_peaks():
    return load("process_memory.csv")


def load_status():
    return load("run_status.csv")


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
             storage_bytes=("memory_bytes", "median"),
             session=("session", "first"))
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
    fig.savefig(path(name), dpi=150, bbox_inches="tight")
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
    """Time / n^3 in nanoseconds per multiply-add, every build: colour = method, line
    style and marker = build. Flat = cubic scaling."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    d0 = select(summary, experiment="dense")
    d0 = d0[d0["method"] != "transpose"]
    for build in BUILD_ORDER:
        for m in DENSE_ORDER:
            d = d0[(d0["method"] == m) & (d0["build"] == build)].sort_values("n")
            if d.empty:
                continue
            scale = 1e9 / d["n"] ** 3   # IQR bar: q1 and q3 in ns per multiply-add
            ax.errorbar(d["n"], d["ns_per_multiply_add"],
                        yerr=[d["ns_per_multiply_add"] - d["q1_time"] * scale,
                              d["q3_time"] * scale - d["ns_per_multiply_add"]],
                        color=COLORS[m], linestyle=BUILD_STYLE[build],
                        marker=BUILD_MARKER[build], markersize=4, linewidth=1.6, capsize=2,
                        elinewidth=1)
    ns = np.sort(d0["n"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time / $n^3$ (ns per multiply-add)", fontsize=11)
    ax.set_title("Cost per operation of the dense kernels", fontsize=12)
    builds = [b for b in BUILD_ORDER if b in set(d0["build"])]
    methods = ax.legend(handles=[Line2D([], [], color=COLORS[m], linewidth=2, label=m)
                                 for m in DENSE_ORDER],
                        title="method", fontsize=9, title_fontsize=9, loc="upper left")
    ax.add_artist(methods)
    ax.legend(handles=[Line2D([], [], color="black", linestyle=BUILD_STYLE[b],
                              marker=BUILD_MARKER[b], markersize=4, label=b) for b in builds],
              title="build", fontsize=9, title_fontsize=9, loc="upper left",
              bbox_to_anchor=(0.24, 1.0))
    add_grid(ax, ns, [str(int(n)) for n in ns])
    save(fig, "plot_cost_per_op.png")


def plot_block_size(summary):
    fig, ax = plt.subplots(figsize=(7, 4.0))
    d0 = select(summary, experiment="blocksize")
    sizes = sorted(d0["n"].unique())
    colors = dict(zip(sizes, ["#4c8eda", "#c0504d", "#5baa5b", "#e07b39", "#7b5ea7"]))
    for build in [b for b in BUILD_ORDER if b in set(d0["build"])]:
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
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
    d0 = select(summary, experiment="density")
    n = int(d0["n"].iloc[0])
    for ax, dist in zip(axes, ["uniform", "band"]):
        dd = d0[d0["distribution"] == dist]
        csr = dd[dd["method"] == "csr"].sort_values("density_value")
        conv = dd[dd["method"] == "convert"].set_index("density")["median_time"]
        errorbar(ax, csr, "density_value", color=COLORS["csr"], label="CSR kernel")
        ax.plot(csr["density_value"], csr["median_time"] + csr["density"].map(conv).values,
                color=COLORS["csr"], linestyle=":", marker="^", markersize=4,
                label="CSR + conversion")
        for build in ["scalar", "avx2"]:
            b = dd[(dd["method"] == "blocked") & (dd["build"] == build)].sort_values("density_value")
            errorbar(ax, b, "density_value", color=COLORS["blocked"], linestyle=BUILD_STYLE[build],
                     label=f"dense blocked ({build})")
        ijk = dd[dd["method"] == "ijk"].sort_values("density_value")
        ax.errorbar(ijk["density_value"], ijk["median_time"],
                    yerr=[ijk["median_time"] - ijk["q1_time"], ijk["q3_time"] - ijk["median_time"]],
                    color=COLORS["ijk"], marker="D", linestyle="none", markersize=5, capsize=3,
                    label="dense ijk (scalar)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Density of A and B (% nonzero, log scale)", fontsize=11)
        ax.set_title(f"{DIST_LABELS[dist]} nonzeros, n = {n}", fontsize=12)
        ds = np.sort(csr["density_value"].unique())
        add_grid(ax, ds, density_ticks(ds))
    axes[0].set_ylabel("Median time (s, log scale)", fontsize=11)
    axes[0].legend(fontsize=9)
    save(fig, "plot_density_time.png")


def plot_output_density(summary, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.3))
    d0 = select(summary, experiment="density", method="csr")
    n = int(d0["n"].iloc[0])
    ds = np.sort(d0["density_value"].unique())
    model = np.geomspace(ds.min(), ds.max(), 200)
    ax.plot(100 * model, 100 * (1 - (1 - model ** 2) ** n), color="gray", linestyle=":",
            linewidth=1, label="uniform model")
    ax.plot(100 * model, 100 * model, color="gray", linestyle="--", linewidth=0.8,
            label="output = input density")
    for dist, color in [("uniform", "#7b5ea7"), ("band", "#e07b39")]:
        d = d0[d0["distribution"] == dist].sort_values("density_value")
        ax.plot(d["density_a_pct"], d["density_c_pct"], marker="o", color=color, linewidth=1.5,
                label=f"{DIST_LABELS[dist]} (measured)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density of A (% nonzero, log scale)", fontsize=12)
    ax.set_ylabel("Density of C = AB (% nonzero, log scale)", fontsize=12)
    ax.set_title(f"Fill-in of the product, n = {n}", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, 100 * ds, [f"{100 * d:g}" for d in ds])
    if fig is not None:
        save(fig, "plot_output_density.png")


def plot_density_memory(summary, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.3))
    d0 = select(summary, experiment="density")
    n = int(d0["n"].iloc[0])
    for dist, color in [("uniform", "#7b5ea7"), ("band", "#e07b39")]:
        d = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")].sort_values("density_value")
        ax.plot(d["density_value"], d["storage_bytes"] / MIB, color=color, marker="o",
                linewidth=1.5, label=f"CSR {dist}: storage")
        ax.plot(d["density_value"], d["peak_bytes"] / MIB, color=color, marker="s",
                linestyle=":", linewidth=1.2, label=f"CSR {dist}: peak")
    b = d0[(d0["method"] == "blocked") & (d0["build"] == "scalar")
           & (d0["distribution"] == "uniform")].sort_values("density_value")
    ax.plot(b["density_value"], b["peak_bytes"] / MIB, color=COLORS["blocked"], marker="s",
            linestyle=":", linewidth=1.2, label="dense blocked: peak")
    ax.axhline(24 * n ** 2 / MIB, color="gray", linestyle="--", linewidth=1,
               label="24$n^2$ bytes")
    ds = np.sort(d0["density_value"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density of A and B (% nonzero, log scale)", fontsize=12)
    ax.set_ylabel("Memory (MiB, log scale)", fontsize=12)
    ax.set_title(f"Memory of CSR and dense, n = {n}", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, ds, density_ticks(ds))
    if fig is not None:
        save(fig, "plot_density_memory.png")


def plot_fill_memory(summary):
    """The two previous figures side by side (the version included in the report)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    plot_output_density(summary, axes[0])
    plot_density_memory(summary, axes[1])
    save(fig, "plot_fill_memory.png")


def plot_size_time(summary, status):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    d0 = select(summary, experiment="size")
    styles = {("uniform", "0.001"): ("#7b5ea7", "-"), ("uniform", "0.01"): ("#7b5ea7", "--"),
              ("band", "0.001"): ("#e07b39", "-"), ("band", "0.01"): ("#e07b39", "--")}
    for (dist, dens), (color, ls) in styles.items():
        d = d0[(d0["method"] == "csr") & (d0["distribution"] == dist) & (d0["density"] == dens)]
        st = select(status, experiment="size", method="csr", distribution=dist, density=dens,
                    process=1)
        note = "".join(f" (n = {r.n}: {r.status})" for r in st.itertuples()
                       if r.status not in ("OK", "SKIPPED"))
        errorbar(ax, d.sort_values("n"), "n", color=color, linestyle=ls,
                 label=f"CSR, {dist}, {100 * float(dens):g} %{note}")
    b = d0[(d0["method"] == "blocked") & (d0["distribution"] == "uniform")
           & (d0["density"] == "0.01")].sort_values("n")
    errorbar(ax, b, "n", color=COLORS["blocked"], label="dense blocked (scalar), uniform 1 %")
    ns = np.sort(select(status, experiment="size")["n"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time (s, log scale)", fontsize=12)
    ax.set_title("Sparse multiplication at fixed density", fontsize=12)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
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
    builds = [b for b in BUILD_ORDER if b in set(d0["build"])]
    for b in sorted(d0["block"].unique()):
        cells = [time_cell(summary, None, experiment="blocksize", build=build, n=n, block=b)
                 for build in builds for n in sorted(d0["n"].unique())]
        lines.append(f"{b} & " + " & ".join(cells) + " \\\\")
    write_table("block_size.tex", lines)


def max_iqr(summary, **kw):
    return select(summary, **kw)["iqr_over_median_pct"].max()


def table_density(summary):
    """Per distribution and density: measured densities of A, B and C, times and ratios; a last row
    gives the number of runs and the largest IQR of each time column."""
    lines = []
    d0 = select(summary, experiment="density")
    for dist in ["uniform", "band"]:
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{DIST_LABELS[dist]} nonzeros}}}} \\\\")
        dd = d0[d0["distribution"] == dist]
        for dens in sorted(dd["density"].unique(), key=float):
            kw = dict(experiment="density", distribution=dist, density=dens)
            csr = one(summary, "median_time", method="csr", **kw)
            conv = one(summary, "median_time", method="convert", **kw)
            bs = one(summary, "median_time", method="blocked", build="scalar", **kw)
            bv = one(summary, "median_time", method="blocked", build="avx2", **kw)
            cells = [f"{100 * float(dens):g}",
                     fmt_pct(one(summary, "density_a_pct", method="csr", **kw)),
                     fmt_pct(100 * one(summary, "nnz_b", method="csr", **kw)
                             / one(summary, "n", method="csr", **kw) ** 2),
                     fmt_pct(one(summary, "density_c_pct", method="csr", **kw)),
                     fmt_time(csr), fmt_time(conv), fmt_time(bs), fmt_time(bv),
                     fmt_sig(bs / csr, 3), fmt_sig(bs / (csr + conv), 3)]
            lines.append(" & ".join(cells) + " \\\\")
        if dist == "uniform":
            lines.append("\\addlinespace")
    kw = dict(experiment="density")
    runs = sorted(set(select(d0, method="csr")["runs"]) | set(select(d0, method="convert")["runs"])
                  | set(select(d0, method="blocked")["runs"]))
    lines += ["\\midrule",
              f"\\multicolumn{{4}}{{l}}{{Largest IQR (\\%)}} & "
              f"{max_iqr(summary, method='csr', **kw):.1f} & "
              f"{max_iqr(summary, method='convert', **kw):.1f} & "
              f"{max_iqr(summary, method='blocked', build='scalar', **kw):.1f} & "
              f"{max_iqr(summary, method='blocked', build='avx2', **kw):.1f} & "
              f"\\multicolumn{{2}}{{r}}{{runs: {', '.join(map(str, runs))}}} \\\\"]
    write_table("density.tex", lines)


def fmt_mib(b):
    if b is None or pd.isna(b):
        return "---"
    x = b / MIB
    return f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "\\,")


SIZE_SERIES = [("uniform", "0.001"), ("uniform", "0.01"), ("band", "0.001"), ("band", "0.01")]


def needed_bytes(detail):
    """Storage the program needed when it stopped with MEMORY_LIMIT ("needs X bytes")."""
    if isinstance(detail, str) and "needs " in detail:
        return int(detail.split("needs ")[1].split()[0])
    return None


def table_size(summary, status):
    """Rows n; per input series the CSR median time and the computed storage (MiB). A
    process that stopped shows its status and, for MEMORY_LIMIT, the storage it needed."""
    lines = []
    st = select(status, experiment="size", method="csr")
    for n in sorted(st["n"].unique()):
        cells = [str(n)]
        for dist, dens in SIZE_SERIES:
            kw = dict(experiment="size", method="csr", distribution=dist, density=dens, n=n)
            t = one(summary, "median_time", **kw)
            if t is not None:
                cells += [fmt_time(t), fmt_mib(one(summary, "storage_bytes", **kw))]
                continue
            s1 = select(status, process=1, **kw)
            if s1.empty:
                cells += ["---", "---"]
                continue
            need = needed_bytes(s1["detail"].iloc[0])
            cells += [s1["status"].iloc[0].replace("_", "\\_"),
                      "---" if need is None else f"({fmt_mib(need)})"]
        cells.append(fmt_time(one(summary, "median_time", experiment="size", method="blocked",
                                  distribution="uniform", density="0.01", n=n)))
        lines.append(" & ".join(cells) + " \\\\")
    write_table("size.tex", lines)


def table_vectorisation(summary):
    """Gain of each build step, median / median: scalar -> SSE2, SSE2 -> AVX2 and
    scalar -> AVX2, for ikj and for blocked (each build with its own b*)."""
    lines = []
    for n in sorted(select(summary, experiment="dense", build="scalar", method="ikj")["n"]):
        cells = [str(n)]
        for m in ["ikj", "blocked"]:
            t = {b: one(summary, "median_time", experiment="dense", method=m, build=b, n=n)
                 for b in BUILD_ORDER}
            for b1, b2 in [("scalar", "sse2"), ("sse2", "avx2"), ("scalar", "avx2")]:
                cells.append(fmt_sig(t[b1] / t[b2], 3) if t[b1] and t[b2] else "---")
        lines.append(" & ".join(cells) + " \\\\")
    write_table("vectorisation.tex", lines)


LIMIT_SERIES = ([("dense", "ijk", "scalar", "dense", "1")]
                + [("dense", m, b, "dense", "1") for m in ["ikj", "blocked"] for b in BUILD_ORDER]
                + [("dense", "transposed", b, "dense", "1") for b in ["scalar", "avx2"]]
                + [("size", "csr", "scalar", dist, dens) for dist, dens in SIZE_SERIES])


def limit_rows(summary, status):
    """Per series: largest n with status OK, its longest median and largest peak, the
    largest n attempted and the first n that stopped (and how)."""
    rows = []
    for exp, m, b, dist, dens in LIMIT_SERIES:
        kw = dict(experiment=exp, method=m, build=b, distribution=dist, density=dens)
        st = select(status, process=1, **kw)
        if st.empty:
            continue
        ok = select(summary, **kw)
        bad = st[st["status"] != "OK"].sort_values("n")
        attempted = st[st["status"] != "SKIPPED"]["n"].max()
        first_bad = None if bad.empty else bad.iloc[0]
        rows.append(dict(series=kw, largest_ok=ok["n"].max(), attempted=attempted,
                         longest=ok["median_time"].max(), peak=ok["peak_bytes"].max(),
                         first_bad=None if first_bad is None else (first_bad["n"],
                                                                     first_bad["status"])))
    return rows


def table_limits(summary, status):
    lines = []
    for r in limit_rows(summary, status):
        kw = r["series"]
        name = (f"\\texttt{{{kw['method']}}} ({kw['build']})" if kw["experiment"] == "dense"
                else f"CSR, {kw['distribution']} {100 * float(kw['density']):g}\\%")
        over = ("none" if r["first_bad"] is None
                else f"{r['first_bad'][0]} ({r['first_bad'][1].replace('_', chr(92) + '_')})")
        lines.append(f"{name} & {r['largest_ok']} & {fmt_time(r['longest'])} & "
                     f"{fmt_mib(r['peak'])} & {over} \\\\")
    write_table("limits.tex", lines)


REFERENCE_SOURCE = "dense"   # the drift references repeat configurations of this experiment


def drift_rows(summary):
    rows = []
    for r in select(summary, experiment="reference").itertuples():
        kw = dict(method=r.method, build=r.build, n=r.n, distribution=r.distribution,
                  density=r.density, block=r.block)
        old = select(summary, experiment=REFERENCE_SOURCE, **kw)
        if old.empty:
            continue
        o = old.iloc[0]
        rows.append(dict(method=r.method, build=r.build, block=r.block, n=r.n,
                         old=o.median_time, old_q1=o.q1_time, old_q3=o.q3_time,
                         new=r.median_time, new_q1=r.q1_time, new_q3=r.q3_time,
                         inside=bool(o.q1_time <= r.median_time <= o.q3_time),
                         diff_pct=100 * (r.median_time / o.median_time - 1)))
    return rows


def table_drift(summary):
    lines = []
    for r in drift_rows(summary):
        name = f"\\texttt{{{r['method']}}} ({r['build']}{', b = ' + str(r['block']) if r['block'] else ''})"
        lines.append(f"{name} & {r['old']:.4g} [{r['old_q1']:.4g}, {r['old_q3']:.4g}] & "
                     f"{r['new']:.4g} [{r['new_q1']:.4g}, {r['new_q3']:.4g}] & "
                     f"{'yes' if r['inside'] else 'no'} & {r['diff_pct']:+.1f}\\% \\\\")
    write_table("drift.tex", lines)


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
    plot_fill_memory(summary)
    plot_size_time(summary, status)
    table_dense_runtime(summary, status)
    table_dense_speedup(summary)
    table_block_size(summary)
    table_density(summary)
    table_size(summary, status)
    table_memory_dense(summary)
    table_vectorisation(summary)
    table_limits(summary, status)
    table_drift(summary)
