"""Validates every kernel against NumPy / SciPy on the benchmark inputs, checks that the
kernels agree bit for bit, and checks the checksum and nonzero counts of every run.

Usage (from the repository root, after run_experiment.py --extension has built the
three programs):
    python analysis/check_correctness.py [--sizes 50,257] [--out DIR]

The checksums of results/ and of results/extension/ (if present) are both checked; a
configuration measured in both sessions must give the same checksum in both.

Outputs (in results/, or in DIR):
  correctness.csv   element-wise comparison with NumPy for the verification inputs
  checksums.csv     checksum and nnz of every configuration vs NumPy / SciPy
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EXTENSION = os.path.join(RESULTS, "extension")
OUT = (os.path.abspath(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv
       else RESULTS)
sys.path.insert(0, ROOT)
from run_experiment import BUILDS, exe  # noqa: E402

SEED_A, SEED_B = 42, 43
MASK_SEED_A, MASK_SEED_B = 1042, 1043
VERIFY_SIZES = ([int(x) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]
                if "--sizes" in sys.argv else [50, 257])
VERIFY_INPUTS = [("dense", "1"), ("uniform", "0.05"), ("uniform", "0.3"),
                 ("band", "0.05"), ("band", "0.3")]
VERIFY_BLOCKS = [16, 32, 64, 128, 256]
VERIFY_METHODS = {"scalar": ["ijk", "ikj", "blocked", "transposed", "csr"],
                  "avx2": ["ikj", "blocked", "transposed", "csr"],
                  "sse2": ["ikj", "blocked", "transposed", "csr"]}    # extension build
# Largest output for which the checksum of a run is recomputed with SciPy
MAX_REFERENCE_NNZ = 10 ** 8

# All inputs are non-negative, so every product is non-negative and the sums have no
# cancellation. The relative error of a sum of n such terms is then at most about
# n * u, with u = 2^-53 ~ 1.1e-16 (n = 4000: ~4.4e-13). A relative tolerance of 1e-11
# leaves a margin of more than 20x, while an indexing error changes the result by a
# relative amount of order 1. An element whose reference is 0 has only zero products and
# must be exactly 0.
RTOL = 1e-11


# --------------------------------------------------------------------------- inputs

def splitmix64_at(seed, p):
    """p-th output (0-based) of SplitMix64, same sequence as the C program."""
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + (p.astype(np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def uniform_at(seed, p):
    return (splitmix64_at(seed, p) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def pattern(n, dist, density):
    """Band half-width and keep probability, computed independently of the C program."""
    if dist == "dense":
        return 0, 1.0
    d = float(density)
    if dist == "uniform":
        return 0, d
    target = d * n * n
    w = 0
    while w < n - 1 and n * (2 * w + 1) - w * (w + 1) < target:
        w += 1
    return w, min(1.0, target / (n * (2 * w + 1) - w * (w + 1)))


def generate(n, dist, density, seed, mask_seed, rows_per_chunk=None):
    """The logical input matrix as a SciPy CSR matrix, built in chunks of rows."""
    w, keep = pattern(n, dist, density)
    rows_per_chunk = rows_per_chunk or max(1, 2 ** 24 // n)
    parts = []
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        i, j = np.meshgrid(np.arange(r0, r1, dtype=np.int64), np.arange(n, dtype=np.int64),
                           indexing="ij")
        p = i * n + j
        mask = np.ones(p.shape, dtype=bool)
        if dist == "band":
            mask &= np.abs(i - j) <= w
        if dist != "dense" and keep < 1.0:
            mask &= uniform_at(mask_seed, p) < keep
        values = np.where(mask, uniform_at(seed, p) * 10.0, 0.0)
        parts.append(sp.csr_matrix(values))
    return sp.vstack(parts, format="csr")


def inputs(n, dist, density):
    return (generate(n, dist, density, SEED_A, MASK_SEED_A),
            generate(n, dist, density, SEED_B, MASK_SEED_B))


def compare(result, ref):
    zero = ref == 0
    exact_zeros = bool((result[zero] == 0).all())
    rel = np.abs(result[~zero] - ref[~zero]) / ref[~zero]
    max_rel = rel.max() if rel.size else 0.0
    return np.abs(result - ref).max(), max_rel, exact_zeros and max_rel <= RTOL


# --------------------------------------------------------------------------- checks

def check_against_numpy():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in VERIFY_SIZES:
            for dist, density in VERIFY_INPUTS:
                A, B = inputs(n, dist, density)
                ref = A.toarray() @ B.toarray()
                results = {}
                for build, methods in VERIFY_METHODS.items():
                    for method in methods:
                        for bs in (VERIFY_BLOCKS if method == "blocked" else [0]):
                            name = f"{method}{'/' + str(bs) if bs else ''} ({build})"
                            out = os.path.join(tmp, "result.txt")
                            subprocess.run([exe(build), "verify", method, str(n), dist,
                                            density, str(bs), out], check=True, cwd=ROOT)
                            results[name] = np.loadtxt(out).reshape(n, n)
                            abs_err, rel_err, ok = compare(results[name], ref)
                            rows.append({"test": "vs NumPy", "n": n, "distribution": dist,
                                         "density": density, "method": name,
                                         "max_abs_error": abs_err, "max_rel_error": rel_err,
                                         "tolerance": f"rtol={RTOL}, zeros exact",
                                         "passed": ok})
                base = results["ijk (scalar)"]
                same = all(np.array_equal(r, base) for r in results.values())
                rows.append({"test": "all methods = ijk (bitwise)", "n": n, "distribution": dist,
                             "density": density, "method": f"{len(results)} kernels",
                             "max_abs_error": 0.0 if same else np.nan, "max_rel_error": np.nan,
                             "tolerance": "exact", "passed": same})
                nnz_ref = (A @ B).nnz
                nnz_csr = int(np.count_nonzero(results["csr (scalar)"]))
                rows.append({"test": "nnz of C vs SciPy", "n": n, "distribution": dist,
                             "density": density, "method": "csr (scalar)",
                             "max_abs_error": abs(nnz_csr - nnz_ref), "max_rel_error": np.nan,
                             "tolerance": "exact", "passed": nnz_csr == nnz_ref})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "correctness.csv"), index=False)
    print(df.to_string(index=False))
    return df["passed"].all()


def check_checksums():
    """Every run with a checksum: same value in every run of the configuration, and equal to
    the NumPy / SciPy reference within RTOL; nnz of A, B and C equal to the reference."""
    raws, statuses = [], []
    for session, folder in [("first", RESULTS), ("extension", EXTENSION)]:
        if not os.path.exists(os.path.join(folder, "run_status.csv")):
            continue
        raws.append(pd.read_csv(os.path.join(folder, "raw_measurements.csv"),
                                float_precision="round_trip", dtype={"density": str})
                    .assign(session=session))
        statuses.append(pd.read_csv(os.path.join(folder, "run_status.csv"),
                                    dtype={"density": str}))
    raw, status = pd.concat(raws), pd.concat(statuses)
    rows = []
    keys = ["n", "distribution", "density"]
    for (n, dist, density), d in raw.groupby(keys):
        nnz_c_max = d["nnz_c"].max()
        too_big = nnz_c_max > MAX_REFERENCE_NNZ
        if not too_big:
            A, B = inputs(n, dist, density)
            C = A @ B
            sums = {"product": C.sum(), "transposition": B.sum(),
                    "conversion": A.sum() + B.sum()}
            nnz_ref = {"product": (A.nnz, B.nnz, C.nnz), "transposition": (A.nnz, B.nnz, B.nnz),
                       "conversion": (A.nnz, B.nnz, A.nnz + B.nnz)}
        for (method, build), g in d.groupby(["method", "build"]):
            kind = {"transpose": "transposition", "convert": "conversion"}.get(method, "product")
            values = g["checksum"].dropna().unique()
            nnz = g[["nnz_a", "nnz_b", "nnz_c"]].drop_duplicates()
            row = {"n": n, "distribution": dist, "density": density, "method": method,
                   "build": build, "sessions": "+".join(sorted(g["session"].unique())),
                   "runs": len(g["checksum"].dropna()),
                   "distinct_checksums": len(values),
                   "checksum": repr(float(values[0])) if len(values) else "",
                   "nnz_c": int(nnz["nnz_c"].iloc[0])}
            if too_big:
                row.update(reference="not computed (output too large)", rel_diff_vs_reference=np.nan,
                           nnz_match=np.nan, passed=len(values) == 1 and len(nnz) == 1)
            else:
                ref = sums[kind]
                rel = np.abs(values - ref).max() / ref if len(values) else np.nan
                nnz_ok = len(nnz) == 1 and tuple(int(x) for x in nnz.iloc[0]) == nnz_ref[kind]
                row.update(reference=repr(float(ref)), rel_diff_vs_reference=rel, nnz_match=nnz_ok,
                           passed=bool(len(values) == 1 and rel <= RTOL and nnz_ok))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "checksums.csv"), index=False)
    print(df.to_string(index=False))
    failed = status[status["status"] == "ERROR"]
    if len(failed):
        print(f"\n{len(failed)} processes ended with ERROR")
    return df["passed"].all() and failed.empty


if __name__ == "__main__":
    ok_numpy = check_against_numpy()
    ok_sums = check_checksums()
    print("\nALL CHECKS PASSED" if ok_numpy and ok_sums else "\nSOME CHECKS FAILED")
