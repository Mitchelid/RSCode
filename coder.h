#ifndef _CODER_H_
#define _CODER_H_

#include "matrix.h"

typedef struct _context {
	char* fname;
	int  flen;
} Context;

/**
 * encodeFile - 将context所指定的文件，分割成n块，并编码出m个冗余块
 * decodeFile - 在当前目录下寻找context制定文件的分块，并求出丢失块
 */
void encodeFile(const Context* context, int n, int m);
void decodeFile(const Context* context, int n, int m);

#endif
