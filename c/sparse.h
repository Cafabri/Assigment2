#ifndef SPARSE_H
#define SPARSE_H

#include <stdint.h>

/* Compressed Sparse Row matrix, n x n, column indices sorted within every row. */
typedef struct {
    int n;
    int64_t nnz;
    int64_t *row_ptr;   /* n + 1 offsets into col and val */
    int32_t *col;
    double *val;
} csr_t;

csr_t *csr_alloc(int n, int64_t nnz);
void csr_free(csr_t *M);
int64_t csr_bytes(const csr_t *M);           /* storage of the three arrays */

csr_t *dense_to_csr(const double *D, int n);  /* exact zeros are not stored */
void csr_to_dense(const csr_t *M, double *D); /* D must hold n*n doubles */

/* C = A * B (Gustavson's row-by-row algorithm). Returns NULL if the output would need
   more than limit_bytes of storage; *out_bytes receives the storage of the output. */
csr_t *csr_multiply(const csr_t *A, const csr_t *B, int64_t limit_bytes, int64_t *out_bytes);

/* Work space of csr_multiply besides its output, in bytes */
int64_t csr_multiply_workspace(int n);

#endif
