"""Builds the C benchmark (scalar and AVX2 builds) and runs every configuration under the
declared budget.

Usage (from the repository root):  python run_experiment.py
                                   python run_experiment.py --extension

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

--extension (a second measurement session, 27 October 2026)
runs, in this order, and writes only to results/extension/ (same file names, plus
block_choice.txt; a run that stops can be resumed: the (configuration, process) pairs
that already have a status are not repeated):
  reference  blocked scalar (b = 32), blocked avx2 (b = 256) and ikj scalar at n = 2000,
             repeated to measure the drift between the two sessions
  blocksize  blocked, build sse2 (vectorised, 16-byte vectors), the same b and n
  dense      ikj and blocked (b* of sse2), build sse2, DENSE_SIZES
  size       csr at n = 64000, uniform and band, 0.1 % and 1 %
  dense      ijk scalar at n = 8000 (expected to exceed the time budget)
Options for testing only: --out DIR, --budget-seconds S, --memory-bytes B, --smoke
(the smallest sizes; the defaults are the declared budget and results/extension).
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
          "avx2": CFLAGS + ["-fvect-cost-model=dynamic", "-mavx2"],
          "sse2": CFLAGS + ["-fvect-cost-model=dynamic"]}     # added by --extension
STUDY_BUILDS = ["scalar", "avx2"]                            # the builds of the first run
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

# --extension: added after the first run (see the docstring)
EXTENSION_DIR = os.path.join(RESULTS, "extension")
EXTENSION_BUILD = "sse2"
REFERENCE_N = 2000
REFERENCES = [("blocked", "scalar", 32), ("blocked", "avx2", 256), ("ikj", "scalar", 0)]
EXTENSION_DENSE_METHODS = ["ikj", "blocked"]
LIMIT_CSR_N = [64000]
LIMIT_CSR_INPUTS = [("uniform", "0.01"), ("uniform", "0.001"), ("band", "0.001"),
                    ("band", "0.01")]
LIMIT_IJK_N = [8000]

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


def save_environment(affinity, out=RESULTS, builds=STUDY_BUILDS, protocol=None):
    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"os: {platform.platform()}"] + hardware() + [
        f"affinity of the runner and all benchmark processes: {affinity}",
        power_plan(),
        f"c compiler: {first_line([GCC, '--version'])}",
        *[f"build {name}: {' '.join(BUILDS[name])}" for name in builds],
        f"python: {platform.python_implementation()} {platform.python_version()} (runner and "
        f"analysis only)",
        f"analysis libraries: {library_versions()}",
        "kernel threads: 1",
        f"peak memory metric: {PEAK_METRIC}",
    ] + (protocol or [
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
    ])
    with open(os.path.join(out, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ------------------------------------------------------------------- building

def build(builds=STUDY_BUILDS, out_dir=RESULTS, missed=False):
    """Compiles the builds and records which loops gcc vectorised in each of them (with
    missed=True, also the loops of the dense kernels that gcc reports it could not
    vectorise)."""
    report = []
    for name in builds:
        flags = BUILDS[name]
        cmd = [GCC, *flags, f'-DBUILD_NAME="{name}"', "-fopt-info-vec-optimized",
               "-o", exe(name), *SOURCES]
        out = subprocess.run(cmd, cwd=C_DIR, capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(out.stderr)
        kernels = [l for l in out.stderr.splitlines() if l.startswith(("matmul.c", "sparse.c"))]
        report += [f"== build {name}: gcc {' '.join(flags)}",
                   *(kernels or ["(no loop of matmul.c or sparse.c vectorised)"])]
        if missed:
            out = subprocess.run([GCC, *flags, "-fopt-info-vec-missed", "-c", "-o", os.devnull,
                                  "matmul.c"], cwd=C_DIR, capture_output=True, text=True)
            report += ["  -fopt-info-vec-missed, matmul.c, loops not vectorised (innermost loops: "
                       "line 15 ijk, 24 ikj, 43 blocked, 55 transposed, 61 transpose):",
                       *("  " + l for l in out.stderr.splitlines()
                         if "couldn't vectorize loop" in l)]
        report.append("")
    with open(os.path.join(out_dir, "vectorisation_report.txt"), "w", encoding="utf-8") as f:
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


STATUS_HEADER = ("experiment,method,build,n,distribution,density,block,process,status,detail,"
                 "wall_seconds")
MEMORY_HEADER = "experiment,method,build,n,distribution,density,block,process,peak_bytes,metric"


def raw_key(line):
    """(configuration label, process) of a line of raw_measurements.csv"""
    c = line.split(",")
    return f"{c[17]},{','.join(c[:6])}", c[18]


def status_key(line):
    """(configuration label, process) of a line of run_status.csv or process_memory.csv"""
    c = line.split(",")
    return ",".join(c[:7]), c[7]


class Runner:
    def __init__(self, out=RESULTS, resume=False):
        self.out = out
        self.done = {}      # (label, process) -> status of the processes already run (resume)
        self.times = {}     # (label, process) -> their measured times
        files = {"raw_measurements.csv": HEADER, "run_status.csv": STATUS_HEADER,
                 "process_memory.csv": MEMORY_HEADER}
        resume = resume and os.path.exists(os.path.join(out, "run_status.csv"))
        if resume:
            self.resume()
        self.raw, self.status, self.memory = (
            open(os.path.join(out, name), "a" if resume else "w", newline="") for name in files)
        if not resume:
            for f, header in zip((self.raw, self.status, self.memory), files.values()):
                f.write(header + "\n")
        self.medians = {}

    def resume(self):
        """Reads the processes that already have a status, and removes the raw and memory
        lines of a process that was interrupted before its status was written."""
        with open(os.path.join(self.out, "run_status.csv")) as f:
            for line in f.read().splitlines()[1:]:
                self.done[status_key(line)] = line.split(",")[8]
        dropped = 0
        for name, header, key in [("raw_measurements.csv", HEADER, raw_key),
                                  ("process_memory.csv", MEMORY_HEADER, status_key)]:
            lines = []
            if os.path.exists(os.path.join(self.out, name)):
                with open(os.path.join(self.out, name)) as f:
                    lines = [l for l in f.read().splitlines()[1:] if l]
            kept = [l for l in lines if key(l) in self.done]
            dropped += len(lines) - len(kept)
            with open(os.path.join(self.out, name), "w", newline="") as f:
                f.write("\n".join([header] + kept) + "\n")
            if name == "raw_measurements.csv":
                for l in kept:
                    if l.split(",")[8] == "measured":
                        self.times.setdefault(raw_key(l), []).append(float(l.split(",")[11]))
        print(f"resuming: {len(self.done)} processes already have a status; {dropped} lines "
              f"of interrupted processes removed", flush=True)

    def benchmark(self, cfg, process):
        """Runs one process; returns its status and the median of its measured runs."""
        key = (cfg.label(), str(process))
        if key in self.done:            # resumed run: this process already has a status
            times = self.times.get(key)
            median = statistics.median(times) if times else None
            if process == 1:
                self.medians[cfg.label()] = median
            return self.done[key], median
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
        if (cfg.label(), "1") in self.done:
            return
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


def save_patterns(out=RESULTS, shapes=None):
    """Band half-width, keep probability and nnz of A for every sparse input."""
    shapes = shapes or sorted({(DENSITY_N, d) for d in DENSITIES}
                              | {(n, d) for n in SPARSE_SIZES for d in SIZE_DENSITIES})
    with open(os.path.join(out, "patterns.csv"), "w", newline="") as f:
        f.write("n,distribution,density,band_half_width,keep_probability,nnz_a\n")
        for n, d in shapes:
            for dist in DISTRIBUTIONS:
                out = subprocess.run([exe("scalar"), "pattern", str(n), dist, d],
                                     capture_output=True, text=True, check=True)
                f.write(out.stdout)


def choose_block_sizes(runner, builds=STUDY_BUILDS, select_n=BLOCK_SELECT_N):
    """Block size with the lowest median at select_n (all processes), per build."""
    import csv
    with open(os.path.join(runner.out, "raw_measurements.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if r["experiment"] == "blocksize" and r["run_type"] == "measured"
                and int(r["n"]) == select_n]
    choice, lines = {}, []
    for build in builds:
        medians = {b: statistics.median(float(r["time_seconds"]) for r in rows
                                        if r["build"] == build and int(r["block"]) == b)
                   for b in BLOCK_SIZES}
        choice[build] = min(medians, key=medians.get)
        lines.append(f"{build}: b* = {choice[build]}; medians at n = {select_n}: "
                     + ", ".join(f"b={b} {t:.4g} s" for b, t in medians.items()))
    with open(os.path.join(runner.out, "block_choice.txt"), "w", encoding="utf-8") as f:
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
                    for b in STUDY_BUILDS for bs in BLOCK_SIZES for n in BLOCK_SIZES_N])
    best = choose_block_sizes(runner)

    configs = [Config("dense", m, b, n, block=best[b] if m == "blocked" else 0)
               for b in STUDY_BUILDS for m in DENSE_METHODS[b] for n in DENSE_SIZES]
    for dist in DISTRIBUTIONS:
        for d in DENSITIES:
            configs += [Config("density", "csr", "scalar", DENSITY_N, dist, d),
                        Config("density", "convert", "scalar", DENSITY_N, dist, d)]
            configs += [Config("density", "blocked", b, DENSITY_N, dist, d, best[b])
                        for b in STUDY_BUILDS]
            if d in DENSITY_IJK:
                configs.append(Config("density", "ijk", "scalar", DENSITY_N, dist, d))
    for dist in DISTRIBUTIONS:
        for d in SIZE_DENSITIES:
            configs += [Config("size", "csr", "scalar", n, dist, d) for n in SPARSE_SIZES]
            configs += [Config("size", "blocked", "scalar", n, dist, d, best["scalar"])
                        for n in SPARSE_SIZES if n <= SIZE_DENSE_MAX_N]
    runner.run_all(configs)


# ------------------------------------------------------------------- --extension

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on):
    """Stops Windows from sleeping while this thread runs (on) or restores the default."""
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


def extension_sizes(smoke):
    """Sizes of the extension; --smoke uses the smallest ones (a test of the runner only:
    the ijk series is the one that exceeds a budget of a few seconds)."""
    if smoke:
        return dict(reference_n=100, block_n=[50, 100], select_n=100, dense=[50, 100],
                    csr=[100, 200], csr_inputs=[("uniform", "0.5"), ("band", "0.01")],
                    ijk=[1500, 2000])
    return dict(reference_n=REFERENCE_N, block_n=BLOCK_SIZES_N, select_n=BLOCK_SELECT_N,
                dense=DENSE_SIZES, csr=LIMIT_CSR_N, csr_inputs=LIMIT_CSR_INPUTS,
                ijk=LIMIT_IJK_N)


def extension_protocol(z):
    return [
        "session: extension added after a review of the first run (results/extension)",
        f"reference: {REFERENCES} (method, build, block) at n = {z['reference_n']}, "
        f"compared with the first session",
        f"block sizes ({EXTENSION_BUILD}): {BLOCK_SIZES} at n = {z['block_n']}, chosen at "
        f"n = {z['select_n']}",
        f"dense ({EXTENSION_BUILD}): {EXTENSION_DENSE_METHODS} at n = {z['dense']}",
        f"size limit: csr at n = {z['csr']}, inputs {z['csr_inputs']}",
        f"time limit: ijk (scalar) at n = {z['ijk']}",
        f"warm-ups: {WARMUPS}; measured runs per process: {REPETITIONS}",
        f"processes: {PROCESSES} interleaved when the first process has a median below "
        f"{CHEAP_SECONDS} s, otherwise 1",
        "batching: a first call shorter than 10 ms is a separate check call, and the runs "
        "are timed in batches lasting >= 10 ms (time per call reported)",
        f"budget: {RUN_BUDGET_SECONDS:g} s per run, {MEMORY_BUDGET_BYTES / 2**30:.4g} GiB of "
        f"matrix storage per process",
    ]


def main_extension(args):
    global RUN_BUDGET_SECONDS, MEMORY_BUDGET_BYTES
    out = EXTENSION_DIR
    if "--out" in args:
        out = os.path.abspath(args[args.index("--out") + 1])
    if "--budget-seconds" in args:
        RUN_BUDGET_SECONDS = float(args[args.index("--budget-seconds") + 1])
    if "--memory-bytes" in args:
        MEMORY_BUDGET_BYTES = int(args[args.index("--memory-bytes") + 1])
    z = extension_sizes("--smoke" in args)

    os.makedirs(out, exist_ok=True)
    keep_awake(True)
    try:
        start = time.monotonic()
        affinity = pin_to_performance_cores()
        build(list(BUILDS), out, missed=True)
        save_environment(affinity, out, list(BUILDS), extension_protocol(z))
        save_patterns(out, [(n, d) for n in z["csr"] for d in sorted({d for _, d in z["csr_inputs"]})])
        runner = Runner(out, resume=True)

        runner.run_all([Config("reference", m, b, z["reference_n"], block=bs)
                        for m, b, bs in REFERENCES])

        runner.run_all([Config("blocksize", "blocked", EXTENSION_BUILD, n, block=bs)
                        for bs in BLOCK_SIZES for n in z["block_n"]])
        best = choose_block_sizes(runner, [EXTENSION_BUILD], z["select_n"])[EXTENSION_BUILD]
        runner.run_all([Config("dense", m, EXTENSION_BUILD, n, block=best if m == "blocked" else 0)
                        for m in EXTENSION_DENSE_METHODS for n in z["dense"]])

        runner.run_all([Config("size", "csr", "scalar", n, dist, d)
                        for dist, d in z["csr_inputs"] for n in z["csr"]])

        runner.run_all([Config("dense", "ijk", "scalar", n) for n in z["ijk"]])
        print(f"extension finished in {(time.monotonic() - start) / 60:.1f} min", flush=True)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    if "--extension" in sys.argv:
        main_extension(sys.argv)
    else:
        main()
