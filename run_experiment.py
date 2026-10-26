"""Builds the C benchmark (scalar and AVX2 builds) and runs every configuration under the
declared budget.

Usage (from the repository root):  python run_experiment.py

Windows only: the C timer, the processor-affinity and peak-memory calls and the hardware
queries use Windows APIs. The toolchain path below must point to the local installation.

Experiments, run in this order:
  blocksize  blocked kernel, both builds, n in BLOCK_SIZES_N, b in BLOCK_SIZES; the block
             size b* of each build is the one with the lowest median at n = BLOCK_SELECT_N
  dense      Part A: ijk (baseline), ikj, blocked (b*), transposed and the transposition
             alone, on the dense matrices of Assignment 1, n in DENSE_SIZES
  density    Part B: csr, the dense-to-CSR conversion and the dense comparators on the same
             logical matrices, n = DENSITY_N, every density and distribution
  size       Part B: csr (and blocked up to SIZE_DENSE_MAX_N) at fixed densities, n in
             SPARSE_SIZES

Outputs:
  results/raw_measurements.csv    every timed run (warm-up, measured and check runs)
  results/run_status.csv          OK / TIMEOUT / MEMORY_LIMIT / ERROR / SKIPPED per process
  results/process_memory.csv      peak memory of every benchmark process (OS metric)
  results/patterns.csv            band half-width, keep probability and nnz of every input
  results/block_choice.txt        block size chosen for each build
  results/vectorisation_report.txt  gcc -fopt-info-vec output of both builds
  results/environment.txt         hardware, OS, toolchain and protocol
"""
import ctypes
import ctypes.wintypes as wt
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
C_DIR = os.path.join(ROOT, "c")

# Toolchain (adjust this path on another machine)
GCC = r"C:\msys64\mingw64\bin\gcc.exe"
CFLAGS = ["-O2", "-ffp-contract=off"]                      # as in Assignment 1
BUILDS = {"scalar": CFLAGS,
          "avx2": CFLAGS + ["-fvect-cost-model=dynamic", "-mavx2"]}
SOURCES = ["matmul.c", "sparse.c", "benchmark.c"]

# gcc needs its own bin directory on PATH to find the assembler and linker
os.environ["PATH"] = os.path.dirname(GCC) + os.pathsep + os.environ["PATH"]

# Declared before running: sizes, densities, repetitions and budget
BLOCK_SIZES = [16, 32, 64, 128, 256]
BLOCK_SIZES_N = [1000, 2000]
BLOCK_SELECT_N = 2000
DENSE_SIZES = [100, 200, 500, 1000, 1500, 2000, 3000, 4000]
DENSE_METHODS = {"scalar": ["ijk", "ikj", "blocked", "transposed", "transpose"],
                 "avx2": ["ikj", "blocked", "transposed"]}   # ijk is not vectorised
DENSITY_N = 2000
DENSITIES = ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1"]
DISTRIBUTIONS = ["uniform", "band"]
DENSITY_IJK = ["0.001", "0.1", "1"]       # ijk on sparse inputs: a check of its independence
SPARSE_SIZES = [1000, 2000, 4000, 8000, 16000, 32000]
SIZE_DENSITIES = ["0.001", "0.01"]
SIZE_DENSE_MAX_N = 4000

WARMUPS = 2
REPETITIONS = 5                       # measured runs per process
PROCESSES = 3                         # independent processes for cheap configurations
CHEAP_SECONDS = 60                    # cheap = median run of the first process below this
RUN_BUDGET_SECONDS = 15 * 60          # maximum duration of a single run
MEMORY_BUDGET_BYTES = 16 * 2 ** 30    # per process, checked by the program before allocating

HEADER = ("method,build,n,distribution,density,block,dtype,threads,run_type,repetition,"
          "batch_size,time_seconds,nnz_a,nnz_b,nnz_c,memory_bytes,checksum,experiment,process")


def exe(build):
    return os.path.join(C_DIR, f"benchmark_{build}.exe")


# ---------------------------------------------------------------- processor affinity

def performance_cores():
    """Logical processors of the highest efficiency class (P-cores), or None."""
    k = ctypes.WinDLL("kernel32")
    size = ctypes.c_ulong(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = (ctypes.c_ubyte * size.value)()
    if not k.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        return None
    cpus, off = [], 0
    while off < size.value:                 # SYSTEM_CPU_SET_INFORMATION records
        length = int.from_bytes(bytes(buf[off:off + 4]), "little")
        cpus.append((buf[off + 14], buf[off + 18]))   # logical index, efficiency class
        off += length
    best = max(c for _, c in cpus)
    p_cores = sorted(lp for lp, c in cpus if c == best)
    return p_cores if len(p_cores) < len(cpus) else None


def pin_to_performance_cores():
    """Restricts this process (and therefore every child it starts) to the P-cores."""
    cpus = performance_cores()
    if not cpus:
        return "not set (no hybrid CPU detected)"
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), sum(1 << c for c in cpus)):
        return "not set (SetProcessAffinityMask failed)"
    return f"P-cores only, logical processors {cpus[0]}-{cpus[-1]} ({len(cpus)})"


# ------------------------------------------------------------------- peak memory

class _Counters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
        (f, ctypes.c_size_t) for f in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage")]


_psapi = ctypes.WinDLL("psapi")
_psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]
PEAK_METRIC = "peak working set (GetProcessMemoryInfo)"


def wait_with_peak(proc):
    """Waits for the process and returns its peak working set in bytes, read through the
    process handle after the process has exited (the handle keeps the counters available)."""
    proc.wait()
    c = _Counters()
    c.cb = ctypes.sizeof(c)
    ok = _psapi.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize if ok else None


# ------------------------------------------------------------------- environment

def run_text(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True)
    except OSError:
        return ""
    return (out.stdout.decode("oem", "replace") + out.stderr.decode("oem", "replace")).strip()


def first_line(cmd):
    text = run_text(cmd)
    return text.splitlines()[0] if text else "not found"


def hardware():
    ps = ("$p = Get-CimInstance Win32_Processor; $m = Get-CimInstance Win32_ComputerSystem;"
          "\"$($p.Name)|$($p.NumberOfCores)|$($p.NumberOfLogicalProcessors)|"
          "$($p.L2CacheSize)|$($p.L3CacheSize)|$($m.TotalPhysicalMemory)\"")
    name, cores, logical, l2, l3, ram = run_text(
        ["powershell", "-NoProfile", "-Command", ps]).split("|")
    p = performance_cores() or []
    return [f"cpu: {name.strip()}",
            f"cores: {cores} physical, {logical} logical",
            f"hybrid layout: {len(p)} logical processors on P-cores"
            + (f" ({p[0]}-{p[-1]}), {int(logical) - len(p)} on E-cores" if p else ""),
            f"cache: L2 {int(l2) // 1024} MB total, L3 {int(l3) // 1024} MB",
            f"ram: {int(ram) / 2**30:.1f} GiB"]


def power_plan():
    text = run_text(["powercfg", "/getactivescheme"])
    return "power plan: " + (text.split(":", 1)[-1].strip() if text else "unknown")


def library_versions():
    versions = []
    for mod in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            versions.append(f"{mod} {__import__(mod).__version__}")
        except ImportError:
            versions.append(f"{mod} not installed")
    return ", ".join(versions)


def save_environment(affinity):
    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"os: {platform.platform()}"] + hardware() + [
        f"affinity of the runner and all benchmark processes: {affinity}",
        power_plan(),
        f"c compiler: {first_line([GCC, '--version'])}",
        *[f"build {name}: {' '.join(flags)}" for name, flags in BUILDS.items()],
        f"python: {platform.python_implementation()} {platform.python_version()} (runner and "
        f"analysis only)",
        f"analysis libraries: {library_versions()}",
        "kernel threads: 1",
        f"peak memory metric: {PEAK_METRIC}",
        f"block sizes: {BLOCK_SIZES} at n = {BLOCK_SIZES_N}, chosen at n = {BLOCK_SELECT_N}",
        f"dense sizes: {DENSE_SIZES}",
        f"density experiment: n = {DENSITY_N}, densities {DENSITIES}, distributions "
        f"{DISTRIBUTIONS}, ijk at densities {DENSITY_IJK}",
        f"size experiment: n = {SPARSE_SIZES}, densities {SIZE_DENSITIES}, blocked up to "
        f"n = {SIZE_DENSE_MAX_N}",
        f"warm-ups: {WARMUPS}; measured runs per process: {REPETITIONS}",
        f"processes: {PROCESSES} interleaved when the first process has a median below "
        f"{CHEAP_SECONDS} s, otherwise 1",
        "batching: a first call shorter than 10 ms is a separate check call, and the runs "
        "are timed in batches lasting >= 10 ms (time per call reported)",
        f"budget: {RUN_BUDGET_SECONDS} s per run, {MEMORY_BUDGET_BYTES / 2**30:.0f} GiB of "
        f"matrix storage per process",
    ]
    with open(os.path.join(RESULTS, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ------------------------------------------------------------------- building

def build():
    """Compiles both builds and records which loops gcc vectorised in each of them."""
    report = []
    for name, flags in BUILDS.items():
        cmd = [GCC, *flags, f'-DBUILD_NAME="{name}"', "-fopt-info-vec-optimized",
               "-o", exe(name), *SOURCES]
        out = subprocess.run(cmd, cwd=C_DIR, capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(out.stderr)
        kernels = [l for l in out.stderr.splitlines() if l.startswith(("matmul.c", "sparse.c"))]
        report += [f"== build {name}: gcc {' '.join(flags)}",
                   *(kernels or ["(no loop of matmul.c or sparse.c vectorised)"]), ""]
    with open(os.path.join(RESULTS, "vectorisation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report), flush=True)


# ------------------------------------------------------------------- running

def run_process(cmd):
    """Runs one benchmark process, killing it when a single run exceeds the budget.

    Returns (status, detail, wall seconds, output lines, peak memory in bytes)."""
    env = dict(os.environ, MEMORY_LIMIT_BYTES=str(MEMORY_BUDGET_BYTES))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=ROOT, env=env)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                     daemon=True).start()

    out = []
    start = last = time.monotonic()
    while True:
        try:
            line = lines.get(timeout=max(0.0, RUN_BUDGET_SECONDS - (time.monotonic() - last)))
        except queue.Empty:
            proc.kill()
            proc.wait()
            return ("TIMEOUT", f"a run exceeded {RUN_BUDGET_SECONDS} s",
                    time.monotonic() - start, out, None)
        if line is None:
            break
        out.append(line.strip())
        last = time.monotonic()
        print("   ", line.strip(), flush=True)

    peak = wait_with_peak(proc)
    wall = time.monotonic() - start
    detail = proc.stderr.read().strip().replace("\n", " ").replace(",", ";")
    if proc.returncode == 3:
        return "MEMORY_LIMIT", detail, wall, out, peak
    if proc.returncode != 0:
        return "ERROR", f"exit code {proc.returncode}: {detail}", wall, out, peak
    return "OK", "", wall, out, peak


class Config:
    def __init__(self, experiment, method, build, n, dist="dense", density="1", block=0):
        self.experiment, self.method, self.build = experiment, method, build
        self.n, self.dist, self.density, self.block = n, dist, density, block

    def series(self):
        """Configurations that differ only in n; a failure stops the larger sizes."""
        return (self.experiment, self.method, self.build, self.dist, self.density, self.block)

    def label(self):
        return (f"{self.experiment},{self.method},{self.build},{self.n},{self.dist},"
                f"{self.density},{self.block if self.method == 'blocked' else ''}")

    def command(self):
        return [exe(self.build), "run", self.method, str(self.n), self.dist, self.density,
                str(self.block), str(WARMUPS), str(REPETITIONS)]


class Runner:
    def __init__(self):
        self.raw = open(os.path.join(RESULTS, "raw_measurements.csv"), "w", newline="")
        self.status = open(os.path.join(RESULTS, "run_status.csv"), "w", newline="")
        self.memory = open(os.path.join(RESULTS, "process_memory.csv"), "w", newline="")
        self.raw.write(HEADER + "\n")
        self.status.write("experiment,method,build,n,distribution,density,block,process,"
                          "status,detail,wall_seconds\n")
        self.memory.write("experiment,method,build,n,distribution,density,block,process,"
                          "peak_bytes,metric\n")
        self.medians = {}

    def benchmark(self, cfg, process):
        """Runs one process; returns its status and the median of its measured runs."""
        print(f"{cfg.label()} process={process}", flush=True)
        status, detail, wall, out, peak = run_process(cfg.command())
        for line in out:
            self.raw.write(f"{line},{cfg.experiment},{process}\n")
        self.status.write(f"{cfg.label()},{process},{status},{detail},{wall:.1f}\n")
        if peak is not None and status == "OK":
            self.memory.write(f"{cfg.label()},{process},{peak},{PEAK_METRIC}\n")
        for f in (self.raw, self.status, self.memory):
            f.flush()
        times = [float(l.split(",")[11]) for l in out if l.split(",")[8] == "measured"]
        median = statistics.median(times) if times else None
        if process == 1:
            self.medians[cfg.label()] = median
        return status, median

    def skip(self, cfg):
        self.status.write(f"{cfg.label()},1,SKIPPED,larger than a size over budget,\n")
        self.status.flush()

    def run_all(self, configs):
        """First process of every configuration in order (a failure skips the larger sizes
        of its series), then the further processes of the cheap ones, interleaved."""
        failed, cheap = set(), []
        for cfg in configs:
            if cfg.series() in failed:
                self.skip(cfg)
                continue
            status, median = self.benchmark(cfg, 1)
            if status != "OK":
                failed.add(cfg.series())
            elif median is not None and median < CHEAP_SECONDS:
                cheap.append(cfg)
        for process in range(2, PROCESSES + 1):
            for cfg in cheap:
                self.benchmark(cfg, process)


def save_patterns():
    """Band half-width, keep probability and nnz of A for every sparse input."""
    shapes = sorted({(DENSITY_N, d) for d in DENSITIES}
                    | {(n, d) for n in SPARSE_SIZES for d in SIZE_DENSITIES})
    with open(os.path.join(RESULTS, "patterns.csv"), "w", newline="") as f:
        f.write("n,distribution,density,band_half_width,keep_probability,nnz_a\n")
        for n, d in shapes:
            for dist in DISTRIBUTIONS:
                out = subprocess.run([exe("scalar"), "pattern", str(n), dist, d],
                                     capture_output=True, text=True, check=True)
                f.write(out.stdout)


def choose_block_sizes(runner):
    """Block size with the lowest median at BLOCK_SELECT_N (all processes), per build."""
    import csv
    with open(os.path.join(RESULTS, "raw_measurements.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if r["experiment"] == "blocksize" and r["run_type"] == "measured"
                and int(r["n"]) == BLOCK_SELECT_N]
    choice, lines = {}, []
    for build in BUILDS:
        medians = {b: statistics.median(float(r["time_seconds"]) for r in rows
                                        if r["build"] == build and int(r["block"]) == b)
                   for b in BLOCK_SIZES}
        choice[build] = min(medians, key=medians.get)
        lines.append(f"{build}: b* = {choice[build]}; medians at n = {BLOCK_SELECT_N}: "
                     + ", ".join(f"b={b} {t:.4g} s" for b, t in medians.items()))
    with open(os.path.join(RESULTS, "block_choice.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return choice


def main():
    os.makedirs(RESULTS, exist_ok=True)
    affinity = pin_to_performance_cores()
    build()
    save_environment(affinity)
    save_patterns()
    runner = Runner()

    runner.run_all([Config("blocksize", "blocked", b, n, block=bs)
                    for b in BUILDS for bs in BLOCK_SIZES for n in BLOCK_SIZES_N])
    best = choose_block_sizes(runner)

    configs = [Config("dense", m, b, n, block=best[b] if m == "blocked" else 0)
               for b in BUILDS for m in DENSE_METHODS[b] for n in DENSE_SIZES]
    for dist in DISTRIBUTIONS:
        for d in DENSITIES:
            configs += [Config("density", "csr", "scalar", DENSITY_N, dist, d),
                        Config("density", "convert", "scalar", DENSITY_N, dist, d)]
            configs += [Config("density", "blocked", b, DENSITY_N, dist, d, best[b])
                        for b in BUILDS]
            if d in DENSITY_IJK:
                configs.append(Config("density", "ijk", "scalar", DENSITY_N, dist, d))
    for dist in DISTRIBUTIONS:
        for d in SIZE_DENSITIES:
            configs += [Config("size", "csr", "scalar", n, dist, d) for n in SPARSE_SIZES]
            configs += [Config("size", "blocked", "scalar", n, dist, d, best["scalar"])
                        for n in SPARSE_SIZES if n <= SIZE_DENSE_MAX_N]
    runner.run_all(configs)


if __name__ == "__main__":
    main()
