#ifndef _GALOIS8BIT_H_ 
#define _GALOIS8BIT_H_

typedef unsigned char UInt8_t;

/**
 * galoisAdd - �����������м��㲢����A,B֮��
 * galoisSub - �����������м��㲢����A,B֮��
 * galoisMul - �����������м��㲢����A,B֮��
 * galoisDiv - �����������м��㲢����A,B֮��
 * galoisPow - �����������м��㲢����A��B�η�
 * galoisInv - �����������м��㲢����A�ĵ���
 */
UInt8_t galoisAdd(UInt8_t A, UInt8_t B);
UInt8_t galoisSub(UInt8_t A, UInt8_t B);
UInt8_t galoisMul(UInt8_t A, UInt8_t B);
UInt8_t galoisDiv(UInt8_t A, UInt8_t B);
UInt8_t galoisPow(UInt8_t A, UInt8_t B);
UInt8_t galoisInv(UInt8_t A);

void galoisEightBitInit(void);

void galois8_parallel_tableInit(void);

#endif
