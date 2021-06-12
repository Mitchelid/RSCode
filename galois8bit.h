#ifndef _GALOIS8BIT_H_ 
#define _GALOIS8BIT_H_

typedef unsigned char UInt8_t;

/**
 * galoisAdd - 在迦罗瓦域中计算并返回A,B之和
 * galoisSub - 在迦罗瓦域中计算并返回A,B之差
 * galoisMul - 在迦罗瓦域中计算并返回A,B之积
 * galoisDiv - 在迦罗瓦域中计算并返回A,B之商
 * galoisPow - 在迦罗瓦域中计算并返回A的B次方
 * galoisInv - 在迦罗瓦域中计算并返回A的倒数
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
