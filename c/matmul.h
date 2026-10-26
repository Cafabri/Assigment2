#ifndef MATMUL_H
#define MATMUL_H

/* Dense kernels: row-major n x n float64 matrices, C must be zeroed by the caller and
   is accumulated into (C += A * B). */
void multiply_ijk(const double *A, const double *B, double *C, int n);
void multiply_ikj(const double *A, const double *B, double *C, int n);
void multiply_blocked(const double *A, const double *B, double *C, int n, int bs);
void multiply_transposed(const double *A, const double *BT, double *C, int n);

/* BT = B transposed (used by multiply_transposed, timed separately) */
void transpose(const double *B, double *BT, int n);

#endif
