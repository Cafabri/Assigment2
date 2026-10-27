"""Every number quoted in the text of the report, computed from the raw data.

Usage (from any directory):  python analysis/report_numbers.py [results_dir]

Reads the first session (results_dir) and, if present, the extension session
(results_dir/extension), as plot_results.py does.

Writes  report_numbers.txt  growth exponents, cost per multiply-add, speedups, effect of
        the block size and of vectorisation, variability, sparse/dense ratios and
        crossovers, fill-in, memory, the largest feasible sizes and, for the extension,
        the SSE2 build, the drift references and the runs at the budget limits.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import (BUILD_ORDER, DENSE_ORDER, drift_rows, limit_rows,  # noqa: E402
                          load_peaks, load_raw, load_status, needed_bytes, one, path, select,
                          summarise)

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


def dense_numbers(summary, raw_measured):
    dense = select(summary, experiment="dense")
    say("== Part A: growth exponent, least-squares slope of log(median) vs log(n)")
    for build in BUILD_ORDER:
        for m in DENSE_ORDER + ["transpose"]:
            d = dense[(dense["build"] == build) & (dense["method"] == m)]
            if d.empty:
                continue
            parts = [f"{lo}-{hi}: {fitted_exponent(d[d['n'].between(lo, hi)]):.2f}"
                     for lo, hi in [(500, 1500), (1500, 4000), (500, 4000)]
                     if len(d[d["n"].between(lo, hi)]) >= 2]
            say(f"  {m} ({build}): " + ", ".join(parts))

    say("\n== Part A: ns per multiply-add, by n")
    for build in BUILD_ORDER:
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
    for build in BUILD_ORDER:
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
             (("avx2", "ikj"), ("avx2", "blocked")),
             (("scalar", "ikj"), ("sse2", "ikj")),
             (("sse2", "ikj"), ("avx2", "ikj")),
             (("scalar", "blocked"), ("sse2", "blocked")),
             (("sse2", "blocked"), ("avx2", "blocked")),
             (("sse2", "ikj"), ("sse2", "blocked"))]
    for (b1, m1), (b2, m2) in pairs:
        parts = []
        for n in sorted(dense["n"].unique()):
            t1 = one(summary, "median_time", experiment="dense", build=b1, method=m1, n=n)
            t2 = one(summary, "median_time", experiment="dense", build=b2, method=m2, n=n)
            if t1 and t2:
                parts.append(f"{n}: {t1 / t2:.2f}")
        say(f"  {m1} ({b1}) / {m2} ({b2}): " + ", ".join(parts))

    say("\n== Part A: speedup over ijk from the medians of each process (range over the "
        "pairs of process medians: min ijk / max method - max ijk / min method)")
    per_process = raw_measured.groupby(["experiment", "method", "build", "n", "block",
                                        "distribution", "density", "process"])[
        "time_seconds"].median()
    ijk = per_process.loc["dense", "ijk", "scalar"].droplevel([1, 2, 3])
    for n in [1500, 2000, 3000, 4000]:
        if n not in ijk.index.get_level_values(0):
            continue
        base = ijk.loc[n]
        parts = []
        for build in BUILD_ORDER:
            for m in DENSE_ORDER[1:]:
                try:
                    t = per_process.loc["dense", m, build, n].droplevel([0, 1, 2])
                except KeyError:
                    continue
                parts.append(f"{m} ({build}) {base.min() / t.max():.2f}-{base.max() / t.min():.2f}"
                             f" (median {one(summary, 'speedup_vs_ijk', experiment='dense', method=m, build=build, n=n):.2f})")
        say(f"  n={n}: ijk process medians {', '.join(f'{x:.4g}' for x in base.values)} s; "
            + "; ".join(parts))
    d = per_process.loc["density", "ijk", "scalar", 2000].droplevel(0)
    say("  ijk n=2000 in the density experiment, process medians (s) per input: " + "; ".join(
        f"{dist} {dens}: " + ", ".join(f"{x:.4g}" for x in g.values)
        + f" (max/min {g.max() / g.min():.3f})"
        for (dist, dens), g in d.groupby(level=[0, 1])))
    say(f"      all inputs: {d.min():.4g}-{d.max():.4g} s (max/min {d.max() / d.min():.3f}); "
        f"dense experiment process medians {ijk.loc[2000].min():.4g}-{ijk.loc[2000].max():.4g} s")
    dd = select(summary, experiment="density", method="ijk", density="1")
    say("  ijk n=2000 on the dense matrix (d = 100 %) in the density experiment: " + ", ".join(
        f"{r.distribution} {r.median_time:.4g} s" for r in dd.itertuples())
        + f"; dense experiment {one(summary, 'median_time', experiment='dense', method='ijk', n=2000):.4g} s")

    say("\n== Part A: ikj reads one 8-byte element of B per multiply-add: implied rate "
        "8 bytes / (ns per multiply-add)")
    for build in BUILD_ORDER:
        for n in [2000, 4000]:
            ns = one(summary, "ns_per_multiply_add", experiment="dense", method="ikj", build=build,
                     n=n)
            if ns is not None:
                say(f"  ikj ({build}) n={n}: {8 / ns:.1f} GB/s")

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
    for build in [b for b in BUILD_ORDER if b in set(d0["build"])]:
        for n in sorted(d0["n"].unique()):
            d = d0[(d0["build"] == build) & (d0["n"] == n)].sort_values("block")
            if d.empty:
                continue
            best = d.loc[d["median_time"].idxmin()]
            say(f"  {build} n={n}: " + ", ".join(
                f"b={int(r.block)} {r.median_time:.4g} ({r.iqr_over_median_pct:.1f} %)"
                for r in d.itertuples())
                + f"; best b={int(best.block)}, worst/best "
                  f"{d['median_time'].max() / best.median_time:.2f}")
    for folder in ["", "extension"]:
        if os.path.exists(path(folder, "block_choice.txt")):
            with open(path(folder, "block_choice.txt")) as f:
                say("  " + f.read().strip().replace("\n", "\n  "))


def variability(all_summary):
    for session in sorted(set(all_summary["session"])):
        say(f"\n== Variability, session {session}")
        variability_of(all_summary[all_summary["session"] == session])


def variability_of(summary):
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

    say("\n== Density experiment: csr time / (n^3 d^2) ns, range over d >= 20 % and d >= 30 %")
    for dist in ["uniform", "band"]:
        c = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")]
        v = c["median_time"] * 1e9 / (c["n"] ** 3 * c["density_value"] ** 2)
        for lo in [0.2, 0.3]:
            w = v[c["density_value"] >= lo]
            say(f"  {dist}, d >= {100 * lo:g} %: {w.min():.3g}-{w.max():.3g}")
        say(f"  {dist}: maximum {v.max():.3g} at d = {100 * c.loc[v.idxmax(), 'density_value']:g} %")

    say("\n== Density experiment: CSR cost per product (time / n^3 d^2) over the dense cost per "
        "multiply-add at n = 2000, and the crossover of a constant-cost model, "
        "sqrt(dense ns / CSR ns at d = 50 %)")
    for build in BUILD_ORDER:
        dense_ns = one(summary, "ns_per_multiply_add", experiment="dense", method="blocked",
                       build=build, n=2000)
        if dense_ns is None:
            continue
        for dist in ["uniform", "band"]:
            c = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")]
            v = c["median_time"] * 1e9 / (c["n"] ** 3 * c["density_value"] ** 2)
            w = v[c["density_value"] >= 0.2]
            half = v[c["density_value"] == 0.5].iloc[0]
            say(f"  blocked {build} ({dense_ns:.3g} ns), {dist}: CSR / dense for d >= 20 %: "
                f"{w.min() / dense_ns:.2g}-{w.max() / dense_ns:.2g}; model crossover with "
                f"{half:.3g} ns: {100 * np.sqrt(dense_ns / half):.3g} %")

    say("\n== Density experiment: nonzeros per row of C (nnz_c / n) and the sorting rule "
        "(scan of the markers when a row has more than n/16 nonzeros, otherwise qsort)")
    for dist in ["uniform", "band"]:
        c = d0[(d0["distribution"] == dist) & (d0["method"] == "csr")].sort_values("density_value")
        say(f"  {dist} (n/16 = {2000 // 16}): " + ", ".join(
            f"{100 * r.density_value:g}%: {r.nnz_c / r.n:.0f}" for r in c.itertuples()))

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
            exp_all = fitted_exponent(d[d["n"] <= 32000])
            segs = [f"{int(a)}-{int(b)}: {np.log(tb / ta) / np.log(b / a):.2f}"
                    for (a, ta), (b, tb) in zip(d[["n", "median_time"]].values[:-1],
                                                d[["n", "median_time"]].values[1:])]
            say(f"  {dist} {dens}: exponent (n = 1000-32000) {exp_all:.2f}; segments " + ", ".join(segs))
            say("      " + ", ".join(
                f"n={int(r.n)}: {r.median_time:.3g} s (IQR {r.iqr_over_median_pct:.1f} %, "
                f"{r.runs} runs), C {r.density_c_pct:.3g} %, nnzC {int(r.nnz_c)}, "
                f"storage {r.storage_bytes / MIB:.1f} MiB, peak {r.peak_bytes / MIB:.1f} MiB"
                for r in d.itertuples()))
            say("      time / (n^3 d^2) ns: " + ", ".join(
                f"n={int(r.n)}: {r.median_time * 1e9 / (r.n ** 3 * float(dens) ** 2):.3g}"
                for r in d.itertuples()))
            say("      nonzeros per row of C (nnz_c / n; n/16 in parentheses): " + ", ".join(
                f"n={int(r.n)}: {r.nnz_c / r.n:.0f} ({int(r.n) // 16})" for r in d.itertuples()))
            b = d0[(d0["method"] == "blocked") & (d0["distribution"] == dist)
                   & (d0["density"] == dens)].set_index("n")["median_time"]
            say("      blocked / csr: " + ", ".join(
                f"n={n}: {b[n] / d.set_index('n')['median_time'][n]:.3g}" for n in b.index))
    st = select(status, experiment="size")
    bad = st[st["status"] != "OK"]
    say("  non-OK statuses: " + ("none" if bad.empty else "; ".join(
        f"{r.method} {r.distribution} {r.density} n={r.n}: {r.status} {r.detail if isinstance(r.detail, str) else ''}"
        + (f" (= {needed_bytes(r.detail) / 2 ** 30:.1f} GiB)" if needed_bytes(r.detail) else "")
        for r in bad.itertuples())))
    for n in [32000, 64000]:
        t4000 = one(summary, "median_time", experiment="size", method="blocked",
                    distribution="uniform", density="0.01", n=4000)
        say(f"  dense at n={n}: one float64 matrix {8 * n ** 2 / 2 ** 30:.1f} GiB, three "
            f"{24 * n ** 2 / 2 ** 30:.1f} GiB; blocked (scalar) extrapolated as n^3 from "
            f"n=4000: {t4000 * (n / 4000) ** 3 / 3600:.1f} h per run")
    csr = d0[d0["method"] == "csr"]
    gap = (csr["peak_bytes"] - csr["storage_bytes"]) / MIB
    say(f"  csr peak working set - computed storage: {gap.min():.1f}-{gap.max():.1f} MiB")
    say(f"  csr largest IQR/median: {csr['iqr_over_median_pct'].max():.1f} %; runs "
        f"{sorted(set(csr['runs']))}")


def limits(summary, status):
    say("\n== Largest feasible size (largest n with status OK) and first n over budget")
    for r in limit_rows(summary, status):
        kw, bad = r["series"], r["first_bad"]
        over = "none" if bad is None else f"{bad[0]} ({bad[1]})"
        say(f"  {kw['experiment']} {kw['method']} ({kw['build']}) {kw['distribution']} "
            f"{kw['density']}: largest OK n = {r['largest_ok']}, longest median "
            f"{r['longest']:.3g} s, largest peak {r['peak'] / MIB:.1f} MiB, largest attempted "
            f"n = {r['attempted']}, first n over budget: {over}")
    bad = status[status["status"] != "OK"]
    say(f"  all non-OK processes: {len(bad)}")
    for r in bad.itertuples():
        say(f"    {r.experiment} {r.method} {r.build} n={r.n} {r.distribution} {r.density}: "
            f"{r.status}")
    for session in sorted(set(status["session"])):
        st = status[status["session"] == session]
        say(f"  session {session}: largest peak working set "
            f"{summary[summary['session'] == session]['peak_bytes'].max() / MIB:.0f} MiB; "
            f"total wall time of all processes {st['wall_seconds'].sum() / 3600:.2f} h "
            f"({st['wall_seconds'].sum() / 60:.1f} min)")
        for (exp,), g in st.groupby(["experiment"]):
            say(f"      {exp}: {g['wall_seconds'].sum() / 60:.1f} min")


def extension_numbers(summary, status):
    if "extension" not in set(status["session"]):
        return
    say("\n== Extension: drift references (new median vs the median of the first session)")
    for r in drift_rows(summary):
        say(f"  {r['method']} ({r['build']}, b={r['block']}) n={r['n']}: first {r['old']:.4g} s "
            f"[IQR {r['old_q1']:.4g}-{r['old_q3']:.4g}], extension {r['new']:.4g} s "
            f"[IQR {r['new_q1']:.4g}-{r['new_q3']:.4g}]; inside the first IQR: "
            f"{'yes' if r['inside'] else 'no'}; difference {r['diff_pct']:+.1f} %")
    say("\n== Extension: vectorisation, gain of each step (median / median)")
    for m in ["ikj", "blocked"]:
        for b1, b2 in [("scalar", "sse2"), ("sse2", "avx2"), ("scalar", "avx2")]:
            parts = []
            for n in sorted(select(summary, experiment="dense", method=m, build="sse2")["n"]):
                t1 = one(summary, "median_time", experiment="dense", method=m, build=b1, n=n)
                t2 = one(summary, "median_time", experiment="dense", method=m, build=b2, n=n)
                if t1 and t2:
                    parts.append((n, t1 / t2))
            if parts:
                big = [g for n, g in parts if n >= 500]
                say(f"  {m} {b1} -> {b2}: " + ", ".join(f"{n}: {g:.2f}" for n, g in parts)
                    + (f"; n >= 500: {min(big):.2f}-{max(big):.2f}" if big else ""))
    say("  share of the scalar -> AVX2 gain (log) obtained by scalar -> SSE2:")
    for m in ["ikj", "blocked"]:
        parts = []
        for n in sorted(select(summary, experiment="dense", method=m, build="sse2")["n"]):
            t = {b: one(summary, "median_time", experiment="dense", method=m, build=b, n=n)
                 for b in BUILD_ORDER}
            if all(t.values()):
                parts.append(f"{n}: {np.log(t['scalar'] / t['sse2']) / np.log(t['scalar'] / t['avx2']):.2f}")
        say(f"    {m}: " + ", ".join(parts))
    st = status[status["session"] == "extension"]
    say("\n== Extension: every non-OK process")
    for r in st[st["status"] != "OK"].itertuples():
        say(f"  {r.experiment} {r.method} {r.build} n={r.n} {r.distribution} {r.density}: "
            f"{r.status} ({r.detail if isinstance(r.detail, str) else ''}; wall "
            f"{r.wall_seconds} s)")


def correctness_numbers():
    """Summary of results/correctness.csv and results/checksums.csv (check_correctness.py)."""
    if not os.path.exists(path("correctness.csv")):
        return
    c = pd.read_csv(path("correctness.csv"))
    k = pd.read_csv(path("checksums.csv"))
    vs = c[c["test"] == "vs NumPy"]
    bit = c[c["test"].str.startswith("all methods")]
    say("\n== Correctness")
    say(f"  vs NumPy: {len(vs)} comparisons, all passed: {bool(vs['passed'].all())}, largest "
        f"relative error {vs['max_rel_error'].max():.2g}; kernels per input "
        f"{sorted(set(vs.groupby(['n', 'distribution', 'density']).size()))} "
        f"(builds {sorted({m.split('(')[1][:-1] for m in vs['method']})})")
    say(f"  bitwise = ijk (scalar): {len(bit)} inputs, all passed: {bool(bit['passed'].all())}")
    nnz = c[c["test"].str.startswith("nnz")]
    say(f"  nnz of C = SciPy: {len(nnz)} inputs, all passed: {bool(nnz['passed'].all())}")
    ref = k[k["reference"].astype(str).str.startswith("not") == False]  # noqa: E712
    say(f"  checksums: {len(k)} configurations ({(k['sessions'] == 'first').sum()} first session "
        f"only, {(k['sessions'] == 'extension').sum()} extension only, "
        f"{(k['sessions'] == 'extension+first').sum()} both), all passed: "
        f"{bool(k['passed'].all())}, distinct checksums per configuration "
        f"{sorted(set(k['distinct_checksums']))}; largest relative difference to the "
        f"reference {ref['rel_diff_vs_reference'].max():.2g}; without reference (output above "
        f"1e8 nonzeros): {len(k) - len(ref)}")


def main():
    raw = load_raw()
    summary = summarise(raw, load_peaks())
    status = load_status()
    dense_numbers(summary, raw[raw["run_type"] == "measured"])
    blocksize_numbers(summary)
    variability(summary)
    density_numbers(summary)
    size_numbers(summary, status)
    limits(summary, status)
    extension_numbers(summary, status)
    correctness_numbers()
    with open(path("report_numbers.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nSaved {path('report_numbers.txt')}")


if __name__ == "__main__":
    main()
