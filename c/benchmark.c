#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <windows.h>
#include "matmul.h"
#include "sparse.h"

#ifndef BUILD_NAME
#define BUILD_NAME "scalar"
#endif

/* Value streams: the generator and seeds of Assignment 1, so that the dense inputs are
   the same matrices. The mask streams decide which positions are nonzero. */
#define SEED_A 42
#define SEED_B 43
#define MASK_SEED_A 1042
#define MASK_SEED_B 1043
/* A first call shorter than BATCH_MIN_SECONDS makes the runs batched: the batch size is
   doubled until a batch lasts at least BATCH_MIN_SECONDS, and the time per call is
   reported. */
#define BATCH_MIN_SECONDS 0.010
#define DEFAULT_MEMORY_LIMIT (16LL << 30)

/* Monotonic clock in seconds (Windows QueryPerformanceCounter) */
static double now(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/* ------------------------------------------------------------------ input matrices */

/*
 * SplitMix64 of Assignment 1. The generator is a counter: its p-th output (p = 0, 1, ...)
 * is mix(seed + (p + 1) * 0x9E3779B97F4A7C15), so any position of the row-major
 * sequence can be computed directly. Position p = i*n + j of a matrix always gets the
 * same value, whatever the density; the nonzeros of a sparser matrix are a subset of
 * those of a denser one with the same distribution.
 */
static uint64_t splitmix64_at(uint64_t seed, int64_t p) {
    uint64_t z = seed + (uint64_t)(p + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* top 53 bits -> [0, 1) */
static double uniform_at(uint64_t seed, int64_t p) {
    return (double)(splitmix64_at(seed, p) >> 11) * 0x1.0p-53;
}

/* Values in [0, 10), exactly as in Assignment 1 */
static double value_at(uint64_t seed, int64_t p) {
    return uniform_at(seed, p) * 10.0;
}

typedef enum { DIST_DENSE, DIST_UNIFORM, DIST_BAND } dist_t;

/*
 * dense:   every position is nonzero (the matrices of Assignment 1).
 * uniform: every position is nonzero with probability d, independently.
 * band:    only positions with |i - j| <= w can be nonzero, w being the smallest
 *          half-width whose band holds at least d * n^2 positions; inside the band every
 *          position is nonzero with probability d * n^2 / (positions of the band).
 */
typedef struct {
    dist_t dist;
    double density;     /* requested density d (fraction of nonzero entries) */
    int band;           /* half-width w (band only) */
    double p_keep;      /* probability of a nonzero at a candidate position */
} pattern_t;

static int64_t band_positions(int n, int w) {
    return (int64_t)n * (2 * w + 1) - (int64_t)w * (w + 1);
}

static pattern_t make_pattern(const char *dist, double density, int n) {
    pattern_t pt = {DIST_DENSE, 1.0, 0, 1.0};
    if (strcmp(dist, "dense") == 0) return pt;
    pt.density = density;
    pt.p_keep = density;
    if (strcmp(dist, "uniform") == 0) {
        pt.dist = DIST_UNIFORM;
    } else if (strcmp(dist, "band") == 0) {
        pt.dist = DIST_BAND;
        double target = density * (double)n * (double)n;
        int w = 0;
        while (w < n - 1 && (double)band_positions(n, w) < target) w++;
        pt.band = w;
        pt.p_keep = target / (double)band_positions(n, w);
        if (pt.p_keep > 1.0) pt.p_keep = 1.0;
    } else {
        fprintf(stderr, "Unknown distribution %s\n", dist);
        exit(2);
    }
    return pt;
}

/* Value of position (i, j), or 0 if the position is not a nonzero of the pattern */
static double entry(const pattern_t *pt, uint64_t seed, uint64_t mask_seed, int n, int i, int j) {
    int64_t p = (int64_t)i * n + j;
    if (pt->dist == DIST_BAND && abs(i - j) > pt->band) return 0.0;
    if (pt->dist != DIST_DENSE && pt->p_keep < 1.0 && uniform_at(mask_seed, p) >= pt->p_keep)
        return 0.0;
    return value_at(seed, p);
}

/* Columns of row i that can hold a nonzero */
static void candidate_columns(const pattern_t *pt, int n, int i, int *j0, int *j1) {
    *j0 = 0; *j1 = n;
    if (pt->dist == DIST_BAND) {
        *j0 = i - pt->band < 0 ? 0 : i - pt->band;
        *j1 = i + pt->band + 1 > n ? n : i + pt->band + 1;
    }
}

static void fill_dense(double *M, int n, uint64_t seed, uint64_t mask_seed, const pattern_t *pt) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[(int64_t)i * n + j] = entry(pt, seed, mask_seed, n, i, j);
}

/* Generates the same logical matrix as fill_dense directly in CSR (two passes: count,
   then fill), so that large sparse inputs never exist as dense arrays. */
static csr_t *generate_csr(int n, uint64_t seed, uint64_t mask_seed, const pattern_t *pt) {
    int64_t nnz = 0;
    for (int i = 0; i < n; i++) {
        int j0, j1;
        candidate_columns(pt, n, i, &j0, &j1);
        for (int j = j0; j < j1; j++)
            if (entry(pt, seed, mask_seed, n, i, j) != 0.0) nnz++;
    }
    csr_t *M = csr_alloc(n, nnz);
    int64_t t = 0;
    for (int i = 0; i < n; i++) {
        int j0, j1;
        candidate_columns(pt, n, i, &j0, &j1);
        M->row_ptr[i] = t;
        for (int j = j0; j < j1; j++) {
            double v = entry(pt, seed, mask_seed, n, i, j);
            if (v != 0.0) { M->col[t] = j; M->val[t] = v; t++; }
        }
    }
    M->row_ptr[n] = t;
    return M;
}

static double *alloc_matrix(int n) {
    double *M = calloc((size_t)n * n, sizeof(double));
    if (!M) { fprintf(stderr, "Out of memory\n"); exit(2); }
    return M;
}

static int64_t count_nonzeros(const double *M, int n) {
    int64_t c = 0;
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        if (M[p] != 0.0) c++;
    return c;
}

static double sum_dense(const double *M, int n) {
    double s = 0.0;
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        s += M[p];
    return s;
}

/* Row-major sum of the stored values: the same additions as sum_dense without the zeros */
static double sum_csr(const csr_t *M) {
    double s = 0.0;
    for (int64_t t = 0; t < M->nnz; t++)
        s += M->val[t];
    return s;
}

/* ------------------------------------------------------------------ methods */

typedef enum { M_IJK, M_IKJ, M_BLOCKED, M_TRANSPOSED, M_TRANSPOSE, M_CSR, M_CONVERT } method_t;
static const char *METHOD_NAMES[] = {"ijk", "ikj", "blocked", "transposed", "transpose",
                                     "csr", "convert"};

static method_t parse_method(const char *s) {
    for (int m = 0; m <= M_CONVERT; m++)
        if (strcmp(s, METHOD_NAMES[m]) == 0) return (method_t)m;
    fprintf(stderr, "Unknown method %s\n", s);
    exit(2);
}

typedef struct {
    method_t method;
    int n, bs;
    int64_t limit;
    double *A, *B, *C, *BT;     /* dense matrices */
    csr_t *As, *Bs, *Cs;        /* sparse matrices; Cs is the last result */
    csr_t *conv_a, *conv_b;     /* last result of a conversion */
} bench_t;

static void dense_kernel(bench_t *b) {
    switch (b->method) {
    case M_IJK:        multiply_ijk(b->A, b->B, b->C, b->n); break;
    case M_IKJ:        multiply_ikj(b->A, b->B, b->C, b->n); break;
    case M_BLOCKED:    multiply_blocked(b->A, b->B, b->C, b->n, b->bs); break;
    case M_TRANSPOSED: multiply_transposed(b->A, b->BT, b->C, b->n); break;
    default: break;
    }
}

static void memory_failure(int64_t need, int64_t limit) {
    fprintf(stderr, "MEMORY_LIMIT: needs %" PRId64 " bytes, limit %" PRId64 "\n", need, limit);
    exit(3);
}

static csr_t *spgemm(bench_t *b) {
    int64_t fixed = csr_bytes(b->As) + csr_bytes(b->Bs) + csr_multiply_workspace(b->n);
    int64_t out_bytes;
    csr_t *C = csr_multiply(b->As, b->Bs, b->limit - fixed, &out_bytes);
    if (!C) memory_failure(fixed + out_bytes, b->limit);
    return C;
}

/*
 * Times `batch` consecutive calls; the result of the last call is kept for the checksum.
 * Dense kernels: C is zeroed before the batch (outside the timer) and accumulates inside
 * it, so every call does the same n^3 multiply-adds (as in Assignment 1).
 * csr / convert: every call allocates its output; the outputs of all but the last call
 * are freed inside the timer.
 */
static double time_batch(bench_t *b, long batch) {
    double t0, t1;
    switch (b->method) {
    case M_TRANSPOSE:
        t0 = now();
        for (long r = 0; r < batch; r++) transpose(b->B, b->BT, b->n);
        t1 = now();
        break;
    case M_CSR:
        csr_free(b->Cs);
        b->Cs = NULL;
        t0 = now();
        for (long r = 0; r < batch; r++) {
            csr_t *C = spgemm(b);
            if (r < batch - 1) csr_free(C); else b->Cs = C;
        }
        t1 = now();
        break;
    case M_CONVERT:
        csr_free(b->conv_a); csr_free(b->conv_b);
        t0 = now();
        for (long r = 0; r < batch; r++) {
            csr_t *ca = dense_to_csr(b->A, b->n), *cb = dense_to_csr(b->B, b->n);
            if (r < batch - 1) { csr_free(ca); csr_free(cb); }
            else { b->conv_a = ca; b->conv_b = cb; }
        }
        t1 = now();
        break;
    default:
        memset(b->C, 0, (size_t)b->n * b->n * sizeof(double));
        t0 = now();
        for (long r = 0; r < batch; r++) dense_kernel(b);
        t1 = now();
    }
    return t1 - t0;
}

static double result_checksum(const bench_t *b) {
    switch (b->method) {
    case M_TRANSPOSE: return sum_dense(b->BT, b->n);
    case M_CSR:       return sum_csr(b->Cs);
    case M_CONVERT:   return sum_csr(b->conv_a) + sum_csr(b->conv_b);
    default:          return sum_dense(b->C, b->n);
    }
}

/* Creates the inputs of a method outside every timed region. Sparse methods generate
   their inputs directly in CSR; the other methods use dense arrays. */
static void setup(bench_t *b, const char *dist, double density, int64_t *nnz_a, int64_t *nnz_b) {
    int n = b->n;
    int64_t dense = (int64_t)n * n * (int64_t)sizeof(double);
    pattern_t pt = make_pattern(dist, density, n);
    if (b->method == M_CSR) {
        b->As = generate_csr(n, SEED_A, MASK_SEED_A, &pt);
        b->Bs = generate_csr(n, SEED_B, MASK_SEED_B, &pt);
        *nnz_a = b->As->nnz;
        *nnz_b = b->Bs->nnz;
        return;
    }
    int matrices = b->method == M_TRANSPOSED ? 4 : b->method == M_CONVERT ? 2 : 3;
    if (matrices * dense > b->limit) memory_failure(matrices * dense, b->limit);
    b->A = alloc_matrix(n);
    b->B = alloc_matrix(n);
    fill_dense(b->A, n, SEED_A, MASK_SEED_A, &pt);
    fill_dense(b->B, n, SEED_B, MASK_SEED_B, &pt);
    if (b->method == M_TRANSPOSE || b->method == M_TRANSPOSED) b->BT = alloc_matrix(n);
    if (b->method == M_TRANSPOSED) transpose(b->B, b->BT, n);
    if (b->method != M_TRANSPOSE && b->method != M_CONVERT) b->C = alloc_matrix(n);
    *nnz_a = count_nonzeros(b->A, n);
    *nnz_b = count_nonzeros(b->B, n);
}

static void teardown(bench_t *b) {
    free(b->A); free(b->B); free(b->C); free(b->BT);
    csr_free(b->As); csr_free(b->Bs); csr_free(b->Cs);
    csr_free(b->conv_a); csr_free(b->conv_b);
}

/* Storage of the matrices of the method, computed from their sizes (not measured) */
static int64_t storage_bytes(const bench_t *b) {
    int64_t dense = (int64_t)b->n * b->n * (int64_t)sizeof(double);
    switch (b->method) {
    case M_TRANSPOSED: return 4 * dense;
    case M_TRANSPOSE:  return 3 * dense;
    case M_CSR:        return csr_bytes(b->As) + csr_bytes(b->Bs) + csr_bytes(b->Cs)
                              + csr_multiply_workspace(b->n);
    case M_CONVERT:    return 2 * dense + csr_bytes(b->conv_a) + csr_bytes(b->conv_b);
    default:           return 3 * dense;
    }
}

static int64_t output_nnz(const bench_t *b) {
    switch (b->method) {
    case M_TRANSPOSE: return count_nonzeros(b->BT, b->n);
    case M_CSR:       return b->Cs->nnz;
    case M_CONVERT:   return b->conv_a->nnz + b->conv_b->nnz;
    default:          return count_nonzeros(b->C, b->n);
    }
}

static int64_t memory_limit(void) {
    const char *s = getenv("MEMORY_LIMIT_BYTES");
    return s ? strtoll(s, NULL, 10) : DEFAULT_MEMORY_LIMIT;
}

static bench_t make_bench(const char *method, int n, int bs) {
    bench_t b;
    memset(&b, 0, sizeof(b));
    b.method = parse_method(method);
    b.n = n;
    b.bs = bs;
    b.limit = memory_limit();
    return b;
}

/* benchmark run <method> <n> <dist> <density> <block> <warmups> <reps>
   One CSV line per run on stdout. */
static void run(const char *method, int n, const char *dist, const char *density, int bs,
                int warmups, int reps) {
    bench_t b = make_bench(method, n, bs);
    int64_t nnz_a, nnz_b;
    setup(&b, dist, atof(density), &nnz_a, &nnz_b);

    /* first call: a check call if it is short (the runs are then batched), otherwise the
       first warm-up run */
    long batch = 1;
    double first = time_batch(&b, 1);
    int64_t nnz_c = output_nnz(&b);
    int64_t mem = storage_bytes(&b);
    const char *label = b.method == M_BLOCKED ? "%s,%s,%d,%s,%s,%d" : "%s,%s,%d,%s,%s,";
    char prefix[160];
    snprintf(prefix, sizeof(prefix), label, method, BUILD_NAME, n, dist, density, bs);

    int r0 = 1;
    if (first < BATCH_MIN_SECONDS) {
        printf("%s,float64,1,check,1,1,%.9e,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
               ",%.17g\n", prefix, first, nnz_a, nnz_b, nnz_c, mem, result_checksum(&b));
        /* calibration (not recorded) */
        while (time_batch(&b, batch) < BATCH_MIN_SECONDS)
            batch *= 2;
    } else {
        int warm = warmups > 0;
        printf("%s,float64,1,%s,1,1,%.9e,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
               ",%.17g\n", prefix, warm ? "warmup" : "measured", first, nnz_a, nnz_b, nnz_c,
               mem, result_checksum(&b));
        r0 = 2;
    }
    fflush(stdout);

    for (int r = r0; r <= warmups + reps; r++) {
        double per_call = time_batch(&b, batch) / batch;
        int warm = r <= warmups;
        int rep = warm ? r : r - warmups;
        if (batch == 1)
            printf("%s,float64,1,%s,%d,1,%.9e,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                   ",%.17g\n", prefix, warm ? "warmup" : "measured", rep, per_call, nnz_a,
                   nnz_b, nnz_c, mem, result_checksum(&b));
        else
            printf("%s,float64,1,%s,%d,%ld,%.9e,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                   ",\n", prefix, warm ? "warmup" : "measured", rep, batch, per_call, nnz_a,
                   nnz_b, nnz_c, mem);
        fflush(stdout);
    }
    teardown(&b);
}

/* benchmark verify <method> <n> <dist> <density> <block> <file>
   Writes the full result matrix (dense, row-major) for the correctness check. For csr it
   also checks that the CSR generator gives the same matrices as fill_dense + dense_to_csr. */
static void verify(const char *method, int n, const char *dist, const char *density, int bs,
                   const char *path) {
    bench_t b = make_bench(method, n, bs);
    int64_t nnz_a, nnz_b;
    setup(&b, dist, atof(density), &nnz_a, &nnz_b);
    time_batch(&b, 1);

    double *R = alloc_matrix(n);
    if (b.method == M_CSR) {
        csr_to_dense(b.Cs, R);
        pattern_t pt = make_pattern(dist, atof(density), n);
        double *D = alloc_matrix(n);
        const csr_t *gen[2] = {b.As, b.Bs};
        uint64_t seeds[2][2] = {{SEED_A, MASK_SEED_A}, {SEED_B, MASK_SEED_B}};
        for (int m = 0; m < 2; m++) {
            fill_dense(D, n, seeds[m][0], seeds[m][1], &pt);
            csr_t *ref = dense_to_csr(D, n);
            if (ref->nnz != gen[m]->nnz
                || memcmp(ref->row_ptr, gen[m]->row_ptr, (size_t)(n + 1) * sizeof(int64_t))
                || memcmp(ref->col, gen[m]->col, (size_t)ref->nnz * sizeof(int32_t))
                || memcmp(ref->val, gen[m]->val, (size_t)ref->nnz * sizeof(double))) {
                fprintf(stderr, "CSR generator differs from dense_to_csr\n");
                exit(1);
            }
            csr_free(ref);
        }
        free(D);
    } else if (b.method == M_TRANSPOSE || b.method == M_CONVERT) {
        fprintf(stderr, "verify is not defined for %s\n", method);
        exit(2);
    } else {
        memcpy(R, b.C, (size_t)n * n * sizeof(double));
    }

    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(2); }
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        fprintf(f, "%.17g\n", R[p]);
    fclose(f);
    free(R);
    teardown(&b);
}

/* benchmark pattern <n> <dist> <density>: band half-width, keep probability and nnz */
static void pattern_info(int n, const char *dist, const char *density) {
    pattern_t pt = make_pattern(dist, atof(density), n);
    csr_t *A = generate_csr(n, SEED_A, MASK_SEED_A, &pt);
    printf("%d,%s,%s,%d,%.17g,%" PRId64 "\n", n, dist, density, pt.band, pt.p_keep, A->nnz);
    csr_free(A);
}

/* ------------------------------------------------------------------ correctness tests */

static int same(const double *X, const double *Y, int n) {
    return memcmp(X, Y, (size_t)n * n * sizeof(double)) == 0;
}

static void fail(const char *test) {
    fprintf(stderr, "%s failed\n", test);
    exit(1);
}

/* All kernels on one pair of n x n matrices must give exactly the result of ijk */
static void check_kernels(const double *A, const double *B, int n, const double *expected,
                          const char *test) {
    double *C = alloc_matrix(n), *BT = alloc_matrix(n), *R = alloc_matrix(n);
    transpose(B, BT, n);
    for (int k = 0; k < 4; k++) {
        memset(C, 0, (size_t)n * n * sizeof(double));
        switch (k) {
        case 0: multiply_ijk(A, B, C, n); break;
        case 1: multiply_ikj(A, B, C, n); break;
        case 2: multiply_blocked(A, B, C, n, 3); break;
        case 3: multiply_transposed(A, BT, C, n); break;
        }
        if (!same(C, expected, n)) fail(test);
    }
    csr_t *As = dense_to_csr(A, n), *Bs = dense_to_csr(B, n);
    int64_t out_bytes;
    csr_t *Cs = csr_multiply(As, Bs, INT64_MAX, &out_bytes);
    csr_to_dense(Cs, R);
    if (!same(R, expected, n) || Cs->nnz != count_nonzeros(expected, n)) fail(test);
    csr_free(As); csr_free(Bs); csr_free(Cs);
    free(C); free(BT); free(R);
}

static void check_correctness(void) {
    double A[] = {1,2, 3,4};
    double I[] = {1,0, 0,1};
    double Z[] = {0,0, 0,0};
    double B[] = {5,6, 7,8};
    double AB[] = {19,22, 43,50};
    check_kernels(A, I, 2, A, "Identity test");
    check_kernels(A, Z, 2, Z, "Zero matrix test");
    check_kernels(A, B, 2, AB, "Known result test");

    /* 7 x 7 sparse matrices with empty rows; block size 3 leaves partial tiles */
    int n = 7;
    pattern_t pt = make_pattern("uniform", 0.3, n);
    double *X = alloc_matrix(n), *Y = alloc_matrix(n), *ref = alloc_matrix(n);
    fill_dense(X, n, SEED_A, MASK_SEED_A, &pt);
    fill_dense(Y, n, SEED_B, MASK_SEED_B, &pt);
    for (int j = 0; j < n; j++) X[2 * n + j] = 0.0;
    multiply_ijk(X, Y, ref, n);
    check_kernels(X, Y, n, ref, "Sparse 7x7 test");
    free(X); free(Y); free(ref);
}

int main(int argc, char **argv) {
    check_correctness();

    if (argc == 9 && strcmp(argv[1], "run") == 0) {
        run(argv[2], atoi(argv[3]), argv[4], argv[5], atoi(argv[6]), atoi(argv[7]),
            atoi(argv[8]));
    } else if (argc == 8 && strcmp(argv[1], "verify") == 0) {
        verify(argv[2], atoi(argv[3]), argv[4], argv[5], atoi(argv[6]), argv[7]);
    } else if (argc == 5 && strcmp(argv[1], "pattern") == 0) {
        pattern_info(atoi(argv[2]), argv[3], argv[4]);
    } else {
        fprintf(stderr,
                "usage: benchmark run <method> <n> <dist> <density> <block> <warmups> <reps>\n"
                "       benchmark verify <method> <n> <dist> <density> <block> <file>\n"
                "       benchmark pattern <n> <dist> <density>\n"
                "methods: ijk ikj blocked transposed transpose csr convert\n"
                "dist: dense uniform band\n");
        return 2;
    }
    return 0;
}
