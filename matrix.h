#pragma once
#ifndef _MATRIX_H_
#define _MATRIX_H_
#include "galois8bit.h"

#define MAX 64

typedef struct Matrix {
	int		m_row;
	int		m_col;
	UInt8_t m_data[MAX][MAX];
} matrix_t;

/**
 * matrixAdd	- 计算两个矩阵之和，返回加和后的矩阵。
 * matrixSub	- 计算两个矩阵之差，返回相减后的矩阵。
 * matrixMul	- 计算两个矩阵相乘，返回相乘后的矩阵
 * matrixNumMul - 计算矩阵倍乘，返回倍乘后的矩阵。
 * matrixTrans	- 计算矩阵转置，返回转置后的矩阵。
 * matrixGauss	- 高斯约旦消去法计算逆矩阵，返回逆矩阵。
 */
matrix_t matrixAdd(matrix_t* A, matrix_t* B);
matrix_t matrixSub(matrix_t* A, matrix_t* B);
matrix_t matrixMul(matrix_t* A, matrix_t* B);
matrix_t matrixNumMul(matrix_t* A, UInt8_t k);
matrix_t matrixTrans(matrix_t* A);
matrix_t matrixGauss(matrix_t* A);

#endif
