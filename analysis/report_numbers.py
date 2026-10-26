"""Every number quoted in the text of the report, computed from the raw data.

Usage (from any directory):  python analysis/report_numbers.py [results_dir]

Writes  report_numbers.txt  growth exponents, cost per multiply-add, speedups, effect of
        the block size and of vectorisation, variability, sparse/dense ratios and
        crossovers, fill-in, memory and the largest feasible sizes.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import (DENSE_ORDER, load_peaks, load_raw, load_status, one,  # noqa: E402
                          path, select, summarise)

MIB = 2 ** 20
out = []


def say(text=""):
    out.append(text)
    print(text)


def fitted_exponent(d, x="n"):
    return np.polyfit(np.log(d[x]), np.log(d["median_time"]), 1)[0]


def crossover(xs, ratio):
    """Log-interpolated x where ratio crosses 1 (first crossing), or None."""
    xs, ratio = np.asarray(xs, float), np.asarray(ratio, float)
    for k in range(len(xs) - 1):
        r1, r2 = ratio[k], ratio[k + 1]
        if (r1 - 1) * (r2 - 1) <= 0 and r1 != r2:
            lx1, lx2, lr1, lr2 = np.log(xs[k]), np.log(xs[k + 1]), np.log(r1), np.log(r2)
            return float(np.exp(lx1 + (0 - lr1) * (lx2 - lx1) / (lr2 - lr1)))
    return None


def dense_numbers(summary):
    dense = select(summary, experiment="dense")
    say("== Part A: growth exponent, least-squares slope of log(median) vs log(n)")
    for build in ["scalar", "avx2"]:
        for m in DENSE_ORDER + ["transpose"]:
            d = dense[(dense["build"] == build) & (dense["method"] == m)]
            if d.empty:
                continue
            parts = [f"{lo}-{hi}: {fitted_exponent(d[d['n'].between(lo, hi)]):.2f}"
                     for lo, hi in [(500, 1500), (1500, 4000), (500, 4000)]
                     if len(d[d["n"].between(lo, hi)]) >= 2]
            say(f"  {m} ({build}): " + ", ".join(parts))

    say("\n== Part A: ns per multiply-add, by n")
    for build in ["scalar", "avx2"]:
        for m in DENSE_ORDER:
            d = dense[(dense["build"] == build) & (dense["method"] == m)].sort_values("n")
            if d.empty:
                continue
            say(f"  {m} ({build}): " + ", ".join(
                f"{int(r.n)}: {r.ns_per_multiply_add:.3g}" for r in d.itertuples()))
            big = d[d["n"] >= 500]["ns_per_multiply_add"]
            say(f"      n >= 500: min {big.min():.3g}, max {big.max():.3g}, "
                f"max/min {big.max() / big.min():.2f}")

    say("\n== Part A: speedup over ijk (scalar), median_ijk / median_method")
    for build in ["scalar", "avx2"]:
        for m in DENSE_ORDER[1:]:
            d = dense[(dense["build"] == build) & (dense["method"] == m)].sort_values("n")
            if d.empty:
                continue
            say(f"  {m} ({build}): " + ", ".join(
                f"{int(r.n)}: {r.speedup_vs_ijk:.2f}" for r in d.itertuples()))

    say("\n== Part A: ratios between optimised methods (median / median)")
    pairs = [(("scalar", "ikj"), ("scalar", "blocked")),
             (("scalar", "transposed"), ("scalar", "ikj")),
             (("scalar", "ikj"), ("avx2", "ikj")),
             (("scalar", "blocked"), ("avx2", "blocked")),
             (("scalar", "transposed"), ("avx2", "transposed")),
             (("avx2", "ikj"), ("avx2", "blocked"))]
    for (b1, m1), (b2, m2) in pairs:
        parts = []
        for n in sorted(dense["n"].unique()):
            t1 = one(summary, "median_time", experiment="dense", build=b1, method=m1, n=n)
            t2 = one(summary, "median_time", experiment="dense", build=b2, method=m2, n=n)
            if t1 and t2:
                parts.append(f"{n}: {t1 / t2:.2f}")
        say(f"  {m1} ({b1}) / {m2} ({b2}): " + ", ".join(parts))

    say("\n== Part A: transposition alone / transposed kernel, and / ijk")
    for n in sorted(dense["n"].unique()):
        t = one(summary, "median_time", experiment="dense", method="transpose", n=n)
        k = one(summary, "median_time", experiment="dense", build="scalar", method="transposed", n=n)
        b = one(summary, "median_time", experiment="dense", method="ijk", n=n)
        if t and k:
            say(f"  n={n}: transpose {t:.3g} s = {100 * t / k:.3g} % of transposed, "
                f"{100 * t / b:.3g} % of ijk; (transposed + transpose) speedup over ijk "
                f"{b / (k + t):.2f}")

    say("\n== Part A: peak working set (MiB) and computed storage")
    for m in DENSE_ORDER:
        d = dense[(dense["build"] == "scalar") & (dense["method"] == m)].sort_values("n")
        big = d[d["n"] >= 1000]
        slope = np.polyfit(big["n"] ** 2, big["peak_bytes"], 1)[0]
        say(f"  {m}: " + ", ".join(f"{int(r.n)}: {r.peak_bytes / MIB:.1f}" for r in d.itertuples())
            + f"; slope vs n^2 (n >= 1000): {slope:.1f} B/element")


def blocksize_numbers(summary):
    say("\n== Block size: median (s) and IQR %")
    d0 = select(summary, experiment="blocksize")
    for build in ["scalar", "avx2"]:
        for n in sorted(d0["n"].unique()):
            d = d0[(d0["build"] == build) & (d0["n"] == n)].sort_values("block")
            best = d.loc[d["median_time"].idxmin()]
            say(f"  {build} n={n}: " + ", ".join(
                f"b={int(r.block)} {r.median_time:.4g} ({r.iqr_over_median_pct:.1f} %)"
                for r in d.itertuples())
                + f"; best b={int(best.block)}, worst/best "
                  f"{d['median_time'].max() / best.median_time:.2f}")
    with open(path("block_choice.txt")) as f:
        say("  " + f.read().strip().replace("\n", "\n  "))


def variability(summary):
    say("\n== Variability")
    long = summary[summary["median_time"] > 1e-3]
    say(f"  configurations: {len(summary)}; IQR/median > 5 % among medians > 1 ms: " + ", ".join(
        f"{r.experiment}/{r.method}/{r.build}/n={r.n}/{r.distribution}/{r.density}: "
        f"{r.iqr_over_median_pct:.1f} %"
        for r in long[long["iqr_over_median_pct"] > 5].itertuples()))
    say(f"  max IQR/median among medians > 1 ms: {long['iqr_over_median_pct'].max():.2f} %")
    say(f"  share with IQR/median <= 3 %: "
        f"{100 * (summary['iqr_over_median_pct'] <= 3).mean():.1f} %")
    multi = summary[summary["processes"] > 1]
    say(f"  configurations with 3 processes: {len(multi)}, single process: "
        f"{len(summary) - len(multi)}; process spread > 10 %: " + ", ".join(
            f"{r.experiment}/{r.method}/{r.build}/n={r.n}/{r.distribution}/{r.density}: "
            f"{r.process_spread_pct:.1f} %"
            for r in multi[multi["process_spread_pct"] > 10].itertuples()))
    say(f"  runs per configuration: {sorted(set(summary['runs']))}")


def density_numbers(summary):
    d0 = select(summary, experiment="density")
    say("\n== Density experiment (n = 2000): nnz and fill-in")
    for dist in ["uniform", "band"]:
        d = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")].sort_values("density_value")
        say(f"  {dist}: " + ", ".join(
            f"d={r.density}: A {r.density_a_pct:.3g} %, C {r.density_c_pct:.3g} % "
            f"(C/A {r.density_c_pct / r.density_a_pct:.2f})" for r in d.itertuples()))

    say("\n== Density experiment: ratios (dense median / CSR median)")
    for dist in ["uniform", "band"]:
        dd = d0[d0["distribution"] == dist]
        csr = dd[dd["method"] == "csr"].set_index("density_value")["median_time"].sort_index()
        conv = dd[dd["method"] == "convert"].set_index("density_value")["median_time"]
        for build in ["scalar", "avx2"]:
            dense = dd[(dd["method"] == "blocked") & (dd["build"] == build)].set_index(
                "density_value")["median_time"]
            r = dense.reindex(csr.index) / csr
            rc = dense.reindex(csr.index) / (csr + conv.reindex(csr.index))
            say(f"  {dist}, blocked {build} / csr: " + ", ".join(
                f"{100 * x:g}%: {v:.3g}" for x, v in r.items()))
            say(f"  {dist}, blocked {build} / (csr + convert): " + ", ".join(
                f"{100 * x:g}%: {v:.3g}" for x, v in rc.items()))
            c1, c2 = crossover(r.index, r.values), crossover(rc.index, rc.values)
            say(f"      crossover density (log-interpolated): kernel only "
                f"{'none' if c1 is None else f'{100 * c1:.3g} %'}, with conversion "
                f"{'none' if c2 is None else f'{100 * c2:.3g} %'}")
        say(f"  {dist}: conversion / csr kernel: " + ", ".join(
            f"{100 * x:g}%: {conv[x] / csr[x]:.3g}" for x in csr.index))
        say(f"  {dist}: conversion time: " + ", ".join(
            f"{100 * x:g}%: {conv[x]:.3g} s" for x in csr.index))
        say(f"  {dist}: csr ns per multiply-add actually performed is not measured; "
            f"csr time / (n^3 d^2) ns: " + ", ".join(
                f"{100 * x:g}%: {csr[x] * 1e9 / (2000 ** 3 * x ** 2):.3g}" for x in csr.index))

    say("\n== Density experiment: dense methods do not depend on the zeros?")
    for m, build in [("blocked", "scalar"), ("blocked", "avx2"), ("ijk", "scalar")]:
        d = d0[(d0["method"] == m) & (d0["build"] == build)]
        say(f"  {m} ({build}): median range {d['median_time'].min():.4g}-"
            f"{d['median_time'].max():.4g} s (max/min {d['median_time'].max() / d['median_time'].min():.3f})")
    say(f"  dense experiment, same n: ijk {one(summary, 'median_time', experiment='dense', method='ijk', n=2000):.4g} s")

    say("\n== Density experiment: memory")
    n = 2000
    for dist in ["uniform", "band"]:
        d = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")].sort_values("density_value")
        say(f"  {dist} csr storage / 24n^2: " + ", ".join(
            f"{r.density}: {r.storage_bytes / (24 * n ** 2):.3g}" for r in d.itertuples()))
        say(f"  {dist} csr peak (MiB): " + ", ".join(
            f"{r.density}: {r.peak_bytes / MIB:.1f}" for r in d.itertuples()))
        x = d["density_value"].values
        y = d["storage_bytes"].values / (24 * n ** 2)
        c = crossover(x, y)
        say(f"      storage crossover (csr = 24n^2): {'none' if c is None else f'{100 * c:.3g} %'}")
    b = d0[(d0["method"] == "blocked") & (d0["build"] == "scalar")]
    say(f"  dense blocked peak (MiB): {b['peak_bytes'].min() / MIB:.1f}-{b['peak_bytes'].max() / MIB:.1f}")


def size_numbers(summary, status):
    d0 = select(summary, experiment="size")
    say("\n== Size experiment: csr")
    for dist in ["uniform", "band"]:
        for dens in ["0.001", "0.01"]:
            d = d0[(d0["method"] == "csr") & (d0["distribution"] == dist)
                   & (d0["density"] == dens)].sort_values("n")
            if d.empty:
                continue
            exp_all = fitted_exponent(d)
            segs = [f"{int(a)}-{int(b)}: {np.log(tb / ta) / np.log(b / a):.2f}"
                    for (a, ta), (b, tb) in zip(d[["n", "median_time"]].values[:-1],
                                                d[["n", "median_time"]].values[1:])]
            say(f"  {dist} {dens}: exponent {exp_all:.2f}; segments " + ", ".join(segs))
            say("      " + ", ".join(
                f"n={int(r.n)}: {r.median_time:.3g} s, C {r.density_c_pct:.3g} %, nnzC {int(r.nnz_c)}, "
                f"storage {r.storage_bytes / MIB:.0f} MiB, peak {r.peak_bytes / MIB:.0f} MiB"
                for r in d.itertuples()))
            b = d0[(d0["method"] == "blocked") & (d0["distribution"] == dist)
                   & (d0["density"] == dens)].set_index("n")["median_time"]
            say("      blocked / csr: " + ", ".join(
                f"n={n}: {b[n] / d.set_index('n')['median_time'][n]:.3g}" for n in b.index))
    st = select(status, experiment="size")
    bad = st[st["status"] != "OK"]
    say("  non-OK statuses: " + ("none" if bad.empty else "; ".join(
        f"{r.method} {r.distribution} {r.density} n={r.n}: {r.status} {r.detail if isinstance(r.detail, str) else ''}"
        for r in bad.itertuples())))


def limits(summary, status):
    say("\n== Largest tested sizes")
    for (exp, m, b, dist, dens), g in summary.groupby(
            ["experiment", "method", "build", "distribution", "density"]):
        if exp not in ("dense", "size"):
            continue
        say(f"  {exp} {m} ({b}) {dist} {dens}: largest OK n = {g['n'].max()}, "
            f"longest median {g['median_time'].max():.3g} s")
    bad = status[status["status"] != "OK"]
    say(f"  all non-OK processes: {len(bad)}")
    for r in bad.itertuples():
        say(f"    {r.experiment} {r.method} {r.build} n={r.n} {r.distribution} {r.density}: "
            f"{r.status}")
    say(f"  largest peak working set: {summary['peak_bytes'].max() / MIB:.0f} MiB")
    wall = status["wall_seconds"].sum()
    say(f"  total wall time of all processes: {wall / 3600:.2f} h")


def main():
    raw = load_raw()
    summary = summarise(raw, load_peaks())
    status = load_status()
    dense_numbers(summary)
    blocksize_numbers(summary)
    variability(summary)
    density_numbers(summary)
    size_numbers(summary, status)
    limits(summary, status)
    with open(path("report_numbers.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nSaved {path('report_numbers.txt')}")


if __name__ == "__main__":
    main()
