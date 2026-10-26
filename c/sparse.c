#include <stdio.h>
#include <stdlib.h>
#include "sparse.h"

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "Out of memory\n"); exit(2); }
    return p;
}

csr_t *csr_alloc(int n, int64_t nnz) {
    csr_t *M = xmalloc(sizeof(csr_t));
    M->n = n;
    M->nnz = nnz;
    M->row_ptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));
    M->col = xmalloc((size_t)nnz * sizeof(int32_t));
    M->val = xmalloc((size_t)nnz * sizeof(double));
    return M;
}

void csr_free(csr_t *M) {
    if (!M) return;
    free(M->row_ptr); free(M->col); free(M->val); free(M);
}

int64_t csr_bytes(const csr_t *M) {
    return (int64_t)(M->n + 1) * sizeof(int64_t)
         + M->nnz * (int64_t)(sizeof(int32_t) + sizeof(double));
}

/* Two passes over the dense matrix: count the nonzeros, then copy them row by row. */
csr_t *dense_to_csr(const double *D, int n) {
    int64_t nnz = 0;
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        if (D[p] != 0.0) nnz++;
    csr_t *M = csr_alloc(n, nnz);
    int64_t t = 0;
    for (int i = 0; i < n; i++) {
        M->row_ptr[i] = t;
        for (int j = 0; j < n; j++) {
            double v = D[(int64_t)i * n + j];
            if (v != 0.0) { M->col[t] = j; M->val[t] = v; t++; }
        }
    }
    M->row_ptr[n] = t;
    return M;
}

void csr_to_dense(const csr_t *M, double *D) {
    int n = M->n;
    for (int64_t p = 0; p < (int64_t)n * n; p++) D[p] = 0.0;
    for (int i = 0; i < n; i++)
        for (int64_t t = M->row_ptr[i]; t < M->row_ptr[i + 1]; t++)
            D[(int64_t)i * n + M->col[t]] = M->val[t];
}

int64_t csr_multiply_workspace(int n) {
    return (int64_t)n * (sizeof(int32_t) + sizeof(double));   /* marker + accumulator */
}

static int cmp_int32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/*
 * Gustavson's algorithm: row i of C is the sum of the rows k of B scaled by A[i][k],
 * for the nonzeros A[i][k] of row i of A in increasing k. A symbolic pass counts the
 * nonzeros of every row of C (so that C is allocated once with its exact size), and a
 * numeric pass accumulates the row in a dense accumulator of n doubles. marker[j] == i
 * means that column j already holds a value in row i.
 *
 * The products of every C[i][j] are added in increasing k, as in the dense kernels; the
 * dense kernels also add the products with a zero factor, which are exactly +0 and do
 * not change the sum, so the nonzero values equal the dense result bit for bit.
 */
csr_t *csr_multiply(const csr_t *A, const csr_t *B, int64_t limit_bytes, int64_t *out_bytes) {
    int n = A->n;
    int32_t *marker = xmalloc((size_t)n * sizeof(int32_t));
    int64_t *row_ptr = xmalloc((size_t)(n + 1) * sizeof(int64_t));

    /* symbolic pass */
    for (int j = 0; j < n; j++) marker[j] = -1;
    row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        int64_t count = 0;
        for (int64_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            int k = A->col[p];
            for (int64_t q = B->row_ptr[k]; q < B->row_ptr[k + 1]; q++) {
                int j = B->col[q];
                if (marker[j] != i) { marker[j] = i; count++; }
            }
        }
        row_ptr[i + 1] = row_ptr[i] + count;
    }

    int64_t nnz = row_ptr[n];
    *out_bytes = (int64_t)(n + 1) * sizeof(int64_t)
               + nnz * (int64_t)(sizeof(int32_t) + sizeof(double));
    if (*out_bytes > limit_bytes) { free(marker); free(row_ptr); return NULL; }

    csr_t *C = xmalloc(sizeof(csr_t));
    C->n = n;
    C->nnz = nnz;
    C->row_ptr = row_ptr;
    C->col = xmalloc((size_t)nnz * sizeof(int32_t));
    C->val = xmalloc((size_t)nnz * sizeof(double));
    double *acc = xmalloc((size_t)n * sizeof(double));

    /* numeric pass */
    for (int j = 0; j < n; j++) marker[j] = -1;
    for (int i = 0; i < n; i++) {
        int32_t *cols = C->col + row_ptr[i];
        int64_t count = 0;
        for (int64_t p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            int k = A->col[p];
            double a = A->val[p];
            for (int64_t q = B->row_ptr[k]; q < B->row_ptr[k + 1]; q++) {
                int j = B->col[q];
                if (marker[j] != i) { marker[j] = i; cols[count++] = j; acc[j] = a * B->val[q]; }
                else acc[j] += a * B->val[q];
            }
        }
        /* sort the column indices of the row: a scan of the marker array costs n steps,
           sorting costs about count * log2(count) comparisons; take the cheaper one */
        if (count > n / 16) {
            int64_t t = 0;
            for (int j = 0; j < n; j++)
                if (marker[j] == i) cols[t++] = j;
        } else {
            qsort(cols, (size_t)count, sizeof(int32_t), cmp_int32);
        }
        double *vals = C->val + row_ptr[i];
        for (int64_t t = 0; t < count; t++)
            vals[t] = acc[cols[t]];
    }
    free(marker);
    free(acc);
    return C;
}
