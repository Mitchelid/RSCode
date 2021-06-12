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
 * matrixAdd	- ������������֮�ͣ����ؼӺͺ�ľ���
 * matrixSub	- ������������֮����������ľ���
 * matrixMul	- ��������������ˣ�������˺�ľ���
 * matrixNumMul - ������󱶳ˣ����ر��˺�ľ���
 * matrixTrans	- �������ת�ã�����ת�ú�ľ���
 * matrixGauss	- ��˹Լ����ȥ����������󣬷��������
 */
matrix_t matrixAdd(matrix_t* A, matrix_t* B);
matrix_t matrixSub(matrix_t* A, matrix_t* B);
matrix_t matrixMul(matrix_t* A, matrix_t* B);
matrix_t matrixNumMul(matrix_t* A, UInt8_t k);
matrix_t matrixTrans(matrix_t* A);
matrix_t matrixGauss(matrix_t* A);

#endif
