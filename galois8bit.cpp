#include <stdio.h>
#include "galois8bit.h"

#define NUM 8
#define POLYNOMIAL (1<<4|1<<3|1<<2|1)

UInt8_t GaloisValue[1 << NUM];
UInt8_t GaloisIndex[1 << NUM];

UInt8_t parallel_low_table[256][16];
UInt8_t parallel_high_table[256][16];

void galoisEightBitInit(void) {
	int i, j = 1;
	for (i = 0; i < (1 << NUM) - 1; ++i) {
		GaloisValue[i] = j;
		GaloisIndex[GaloisValue[i]] = i;
		if (j & 0x80) {
			j <<= 1;
			j ^= POLYNOMIAL;
		}
		else {
			j <<= 1;
		}
	}
}



UInt8_t galoisAdd(UInt8_t A, UInt8_t B) {
	return A ^ B;
}

UInt8_t galoisSub(UInt8_t A, UInt8_t B) {
	return A ^ B;
}

UInt8_t galoisMul(UInt8_t A, UInt8_t B) {
	if (!A || !B) {
		return 0;
	}
	UInt8_t i = GaloisIndex[A];
	UInt8_t j = GaloisIndex[B];
	UInt8_t index = (i + j) % 255;
	return GaloisValue[index];
}

UInt8_t galoisDiv(UInt8_t A, UInt8_t B) {
	if (!A || !B) {
		return 0;
	}
	return galoisMul(A, galoisInv(B));
}

UInt8_t galoisPow(UInt8_t A, UInt8_t B) {
	if (!A) {
		return 0;
	}
	if (!B) {
		return 1;
	}
	UInt8_t i, r = 1;
	for (i = 0; i < B; ++i) {
		r = galoisMul(r, A);
	}
	return r;
}

UInt8_t galoisInv(UInt8_t A) {
	if (!A) {
		return 0;
	}
	UInt8_t j = GaloisIndex[A];
	return GaloisValue[(255 - j) % 255];
}


void galois8_parallel_tableInit(void){
	int i,j;
	for(i=0;i<256;i++){
		for(j=0;j<16;j++){
			parallel_low_table[i][j]=galoisMul(i,j);
			parallel_high_table[i][j]=galoisMul(i,j<<4);
		}
	}
}


