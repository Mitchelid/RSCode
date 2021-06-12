#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "matrix.h"


matrix_t matrixAdd(matrix_t* A, matrix_t* B) {
	assert(A && B);
	assert(A->m_col == B->m_col);
	assert(A->m_row == B->m_row);

	int i, j;
	matrix_t r;
	UInt8_t tmp;
	r.m_col = A->m_col;
	r.m_row = A->m_row;

	for (i = 0; i < r.m_row; ++i) {
		for (j = 0; j < r.m_col; ++j) {
			tmp = galoisAdd(A->m_data[i][j], B->m_data[i][j]);
			r.m_data[i][j] = tmp;
		}
	}
	return r;
}

matrix_t matrixSub(matrix_t* A, matrix_t* B) {
	assert(A && B);
	assert(A->m_col == B->m_col);
	assert(A->m_row == B->m_row);

	int i, j;
	matrix_t r;
	UInt8_t tmp;
	r.m_col = A->m_col;
	r.m_row = A->m_row;

	for (i = 0; i < r.m_row; ++i) {
		for (j = 0; j < r.m_col; ++j) {
			tmp = galoisSub(A->m_data[i][j], B->m_data[i][j]);
			r.m_data[i][j] = tmp;
		}
	}
	return r;
}

matrix_t matrixMul(matrix_t* A, matrix_t* B) {
	assert(A && B);
	assert(A->m_col == B->m_row);

	int i, j, k;
	UInt8_t sum, tmp;
	matrix_t r;
	r.m_row = A->m_row;
	r.m_col = B->m_col;

	for (i = 0; i < A->m_row; ++i) {
		for (j = 0; j < B->m_col; ++j) {
			sum = 0;
			for (k = 0; k < A->m_col; ++j) {
				tmp = galoisMul(A->m_data[i][k], B->m_data[k][j]);
				sum = galoisAdd(sum, tmp);
			}
			r.m_data[i][j] = sum;
		}
	}
	return r;
}

matrix_t matrixNumMul(matrix_t* A, UInt8_t k) {
	assert(A);

	int i, j;
	matrix_t r;
	r.m_row = A->m_row;
	r.m_col = A->m_col;

	for (i = 0; i < A->m_row; ++i) {
		for (j = 0; j < A->m_col; ++j) {
			r.m_data[i][j] = galoisMul(k, A->m_data[i][j]);
		}
	}
	return r;
}

matrix_t matrixTrans(matrix_t* A) {
	assert(A);

	int i, j;
	matrix_t r;
	r.m_row = A->m_col;
	r.m_col = A->m_row;

	for (i = 0; i < A->m_row; ++i) {
		for (j = 0; j < A->m_col; ++j) {
			r.m_data[j][i] = A->m_data[i][j];
		}
	}
	return r;
}

matrix_t matrixGauss(matrix_t* A) {
	int i, j, k, max, pos, len;
	UInt8_t tmpdiv, tmpmul, tmpinv;
	UInt8_t swaparr[MAX] = { 0 };
	matrix_t src, dest;
	dest.m_row = A->m_row;
	dest.m_col = A->m_col;
	memcpy(&src, A, sizeof(matrix_t));
	memset(dest.m_data, 0, sizeof(UInt8_t) * MAX * MAX);

	for (i = 0; i < dest.m_row; ++i) {
		dest.m_data[i][i] = 1;
	}

	for (k = 0; k < src.m_col; ++k) {
		max = src.m_data[k][k];
		pos = k;
		for (i = k + 1; i < src.m_row; ++i) {
			if (src.m_data[i][k] > max) {
				pos = i;
				max = src.m_data[i][k];
			}
		}
		if (pos != k) {  // swap the pos row and the k row
			len = src.m_col * sizeof(UInt8_t);
			memcpy(swaparr, src.m_data[pos], len);
			memcpy(src.m_data[pos], src.m_data[k], len);
			memcpy(src.m_data[k], swaparr, len);
			memcpy(swaparr, dest.m_data[pos], len);
			memcpy(dest.m_data[pos], dest.m_data[k], len);
			memcpy(dest.m_data[k], swaparr, len);
		}
		for (i = 0; i < src.m_row; ++i) {
			if (i != k) {
				tmpdiv = galoisDiv(src.m_data[i][k], src.m_data[k][k]);
				for (j = 0; j < src.m_col; ++j) {
					tmpmul = galoisMul(tmpdiv, src.m_data[k][j]);
					src.m_data[i][j] = galoisAdd(tmpmul, src.m_data[i][j]);
					tmpmul = galoisMul(tmpdiv, dest.m_data[k][j]);
					dest.m_data[i][j] = galoisAdd(tmpmul, dest.m_data[i][j]);
				}
			}
		}
	}
	for (i = 0; i < dest.m_row; ++i) {
		tmpinv = galoisInv(src.m_data[i][i]);
		for (j = 0; j < dest.m_col; ++j) {
			dest.m_data[i][j] = galoisMul(dest.m_data[i][j], tmpinv);
		}
	}
	return dest;
}

void matrixDisPlay(matrix_t* A) {
	int i, j;
	for (i = 0; i < A->m_row; ++i) {
		for (j = 0; j < A->m_col; ++j) {
			printf("%03d ", A->m_data[i][j]);
		}
		printf("\n");
	}
	printf("===================================\n");
}
