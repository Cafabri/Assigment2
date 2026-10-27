# Assignment 2 — Optimised Dense and Sparse Matrix Multiplication in C

**BIG DATA · Academic year 2026-2027**  
Grado en Ciencia e Ingeniería de Datos — Universidad de Las Palmas de Gran Canaria  
**Author:** Carlos Falcón Brito

## Research questions

1. How do loop reordering, cache blocking and a transposed operand change the runtime and
   memory of the triple-loop dense multiplication of Assignment 1, and how much of the
   change comes from compiler vectorisation?
2. Under which tested densities and nonzero distributions does a CSR sparse multiplication
   need less time or memory than a dense method on the same logical matrices?

## Reused material

The baseline is the C implementation of Assignment 1
([Cafabri/Assigment1](https://github.com/Cafabri/Assigment1), commit `0785887`):
`multiply_ijk` in `c/matmul.c` is its kernel with only the function renamed, and the
SplitMix64 generator, the seeds (42 and 43), the timer, the batching rule, the processor
affinity and the peak-memory measurement are the same. The dense inputs are therefore the
matrices of Assignment 1, and the checksums of the baseline match those of Assignment 1.

## Platform

**The study runs on Windows only.** The C timer (`QueryPerformanceCounter`), the processor
affinity (`SetProcessAffinityMask`), the peak-memory measurement (`GetProcessMemoryInfo`)
and the hardware queries (PowerShell, `powercfg`) use Windows APIs. Running the study on
another Windows machine reproduces the procedure, not the numbers.

## Repository structure

```
├── run_experiment.py          # Builds the binaries, runs every configuration under the budget
├── c/
│   ├── matmul.h / matmul.c    # Dense kernels: ijk (baseline), ikj, blocked, transposed
│   ├── sparse.h / sparse.c    # CSR format, dense <-> CSR conversion, Gustavson SpGEMM
│   └── benchmark.c            # Input generation, timing, batching, checksum, verify mode
├── analysis/
│   ├── check_correctness.py   # Comparison with NumPy / SciPy, bitwise and checksum checks
│   ├── plot_results.py        # summary.csv, figures and the LaTeX tables of the report
│   └── report_numbers.py      # Every number quoted in the text of the report
├── results/
│   ├── raw_measurements.csv   # Every timed run (warm-up, measured and check runs)
│   ├── run_status.csv         # OK / TIMEOUT / MEMORY_LIMIT / ERROR / SKIPPED per process
│   ├── process_memory.csv     # Peak working set of every benchmark process
│   ├── patterns.csv           # Band half-width, keep probability and nnz of every input
│   ├── block_choice.txt       # Block size selected for each build
│   ├── vectorisation_report.txt  # Loops vectorised by gcc in each build (-fopt-info-vec)
│   ├── environment.txt        # Hardware, OS, toolchain, affinity, power plan, protocol
│   ├── correctness.csv        # Element-wise comparison with NumPy / SciPy
│   ├── checksums.csv          # Checksum and nnz of every configuration vs the reference
│   ├── summary.csv            # Median, IQR, process spread, ns per multiply-add, memory
│   ├── report_numbers.txt     # Ratios, crossovers and fits quoted in the report
│   ├── tables/*.tex           # Generated tables; the report includes block_size, dense_runtime,
│   │                          #   vectorisation, density, size and limits
│   ├── plot_*.png             # Generated figures; the report uses plot_cost_per_op,
│   │                          #   plot_fill_memory and plot_density_time
│   └── extension/             # Second session (run_experiment.py --extension): the same
│                              #   raw_measurements, run_status, process_memory, patterns,
│                              #   block_choice, vectorisation_report (3 builds, with the loops
│                              #   gcc could not vectorise) and environment files
└── report/
    ├── Assignment2_Report.tex
    └── Assignment02_FalconBrito_Report.pdf
```

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro |
| CPU | Intel Core i7-13700KF: 8 P-cores (logical processors 0–15) + 8 E-cores (16–23) |
| Cache / RAM | 48 KB L1D and 2 MB L2 per P-core, 30 MB shared L3 / 31.8 GiB |
| Affinity | runner and all benchmark processes restricted to the P-cores (0–15) |
| Power plan | Balanced (Windows default) |
| C compiler | gcc 16.1.0 (MSYS2/MinGW64) |
| Build `scalar` | `-O2 -ffp-contract=off` (the flags of Assignment 1; no loop is vectorised) |
| Build `avx2` | `-O2 -ffp-contract=off -fvect-cost-model=dynamic -mavx2` |
| Build `sse2` | `-O2 -ffp-contract=off -fvect-cost-model=dynamic` (extension; 16-byte vectors) |
| Python | CPython 3.11.9 (runner and analysis only) |
| Analysis | numpy 2.2, scipy 1.15, pandas 2.2, matplotlib 3.10 |

## Reproducing the study

Requirements: MSYS2 with `mingw-w64-x86_64-gcc`, Python 3.11 and
`pip install numpy scipy pandas matplotlib`. The path to `gcc.exe` is set at the top of
`run_experiment.py` (`GCC`); change it if gcc is installed elsewhere.

```bash
python run_experiment.py                # build + all benchmarks (3.3 h of benchmark processes here)
python run_experiment.py --extension    # extension, writes results/extension/ (46.2 min here)
python analysis/check_correctness.py    # validation against NumPy / SciPy + checksums (both sessions)
python analysis/plot_results.py         # summary.csv, figures and tables
python analysis/report_numbers.py       # numbers quoted in the text
```

To compile the report (run twice so that references resolve):

```bash
cd report
pdflatex -jobname=Assignment02_FalconBrito_Report Assignment2_Report.tex
```

The benchmark can also be built and run on its own:

```bash
cd c
gcc -O2 -ffp-contract=off -DBUILD_NAME='"scalar"' -o benchmark_scalar.exe matmul.c sparse.c benchmark.c
gcc -O2 -ffp-contract=off -fvect-cost-model=dynamic -mavx2 -DBUILD_NAME='"avx2"' -o benchmark_avx2.exe matmul.c sparse.c benchmark.c
gcc -O2 -ffp-contract=off -fvect-cost-model=dynamic -DBUILD_NAME='"sse2"' -o benchmark_sse2.exe matmul.c sparse.c benchmark.c
benchmark_scalar.exe run ikj 1000 dense 1 0 2 5          # method n dist density block warm-ups runs
benchmark_scalar.exe run blocked 1000 dense 1 32 2 5
benchmark_scalar.exe run csr 2000 uniform 0.01 0 2 5
benchmark_scalar.exe run convert 2000 band 0.01 0 2 5     # dense -> CSR of A and B only
benchmark_scalar.exe verify csr 50 band 0.3 0 out.txt     # writes the full result matrix
benchmark_scalar.exe pattern 2000 band 0.01               # band half-width, keep probability, nnz
```

Methods: `ijk`, `ikj`, `blocked`, `transposed`, `transpose` (the transposition alone),
`csr`, `convert`. Distributions: `dense`, `uniform`, `band`.

## Experimental settings

- **Inputs:** values from the SplitMix64 generator of Assignment 1 (seed 42 for A, 43 for
  B, uniform in [0, 10), position p = i·n + j). The generator is a counter, so position p
  always gets the same value. Nonzero positions: `dense` keeps every position; `uniform`
  keeps a position when a second SplitMix64 stream (seeds 1042 and 1043) gives a uniform
  number below the density d; `band` only allows |i − j| ≤ w, with w the smallest
  half-width whose band holds d·n² positions, and keeps each band position with
  probability d·n² / (band positions). `check_correctness.py` regenerates every input
  independently with NumPy.
- **Experiments:** `blocksize` (b ∈ {16, 32, 64, 128, 256}, n ∈ {1000, 2000}, the block
  size of each build is the one with the lowest median at n = 2000); `dense` (Part A,
  n ∈ {100, 200, 500, 1000, 1500, 2000, 3000, 4000}); `density` (Part B, n = 2000,
  d ∈ {0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100} %, uniform and band); `size` (Part B,
  CSR for n ∈ {1000, …, 32000} at d ∈ {0.1, 1} %, dense blocked up to n = 4000).
- **Extension** (`--extension`, a second measurement session on 2026-10-27,
  46.2 min of benchmark processes): drift references (`reference`: blocked scalar
  b = 32, blocked AVX2 b = 256 and ikj scalar at n = 2000, compared with the first
  session); the `sse2` build (block sweep and ikj / blocked at every dense size); CSR at
  n = 64000 for the four size-experiment inputs; ijk scalar at n = 8000 (beyond the time
  budget). It writes only to `results/extension/` and can be resumed: the processes that
  already have a status are not repeated. `--out`, `--budget-seconds`, `--memory-bytes`
  and `--smoke` exist only to test the runner.
- **Timer:** only the kernel call. Dense kernels: inputs and zeroed result prepared
  outside the timer (as in Assignment 1); `transposed` receives Bᵀ prepared outside the
  timer, and the transposition is timed separately (`transpose`). CSR: inputs already in
  CSR; the timer includes the allocation and sorting of the output. The dense → CSR
  conversion of A and B is timed separately (`convert`).
- **Batching:** a first call shorter than 10 ms is a separate check call, and the runs are
  timed in batches lasting at least 10 ms; the time per call is reported.
- **Warm-up and repetitions:** 2 warm-up runs and 5 measured runs per process;
  configurations whose first process has a median below 60 s run in 3 independent
  processes, interleaved (15 runs).
- **Budget:** 15 minutes per individual run (the process is killed: TIMEOUT) and 16 GiB of
  matrix storage per process, checked by the program before allocating (MEMORY_LIMIT).
  Larger sizes of a failed series are recorded as SKIPPED.
- **Memory:** peak working set of every timing process (measured), and the storage of the
  matrices of the method (computed: 24n² bytes for the dense kernels, 32n² with Bᵀ;
  8(n + 1) + 12·nnz bytes per CSR matrix plus 12n bytes of work space for the SpGEMM).

## Raw data format (`results/raw_measurements.csv`)

| Column | Description |
|--------|-------------|
| `method` | `ijk`, `ikj`, `blocked`, `transposed`, `transpose`, `csr` or `convert` |
| `build` | `scalar`, `avx2` or `sse2` |
| `n` | Matrix side length |
| `distribution` | `dense`, `uniform` or `band` |
| `density` | Requested density d of A and B (fraction of nonzero entries) |
| `block` | Block size b (blocked only) |
| `dtype` | Numeric type (`float64`) |
| `threads` | Threads used by the kernel (1) |
| `run_type` | `warmup`, `measured` or `check` (separate first call of a batched configuration) |
| `repetition` | Run index within its type |
| `batch_size` | Calls per timed batch |
| `time_seconds` | Wall-clock time per call (batch time / batch size) |
| `nnz_a`, `nnz_b`, `nnz_c` | Nonzeros of A, B and of the output (for `convert`: nnz(A) + nnz(B)) |
| `memory_bytes` | Computed storage of the matrices of the method |
| `checksum` | Sum of all elements of the output (empty for batched runs) |
| `experiment` | `blocksize`, `dense`, `density`, `size` or `reference` (extension) |
| `process` | Index of the independent process (1–3) |

## Use of generative AI

Claude Opus 5.5 (Anthropic), used through Claude Code, served as a support tool during this
assignment: to help write and debug the benchmark program, the experiment runner and the analysis scripts,
including those of the second measurement session, and to improve the structure and wording of
the report. All the measurements come from running the code on the machine described in
Section 3.3 of the report. The author reviewed
the code and the AI-generated text, checked the numbers against the raw data and the
references, and is responsible for the conclusions.
