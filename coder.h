#ifndef _CODER_H_
#define _CODER_H_

#include "matrix.h"

typedef struct _context {
	char* fname;
	int  flen;
} Context;

/**
 * encodeFile - ��context��ָ�����ļ����ָ��n�飬�������m�������
 * decodeFile - �ڵ�ǰĿ¼��Ѱ��context�ƶ��ļ��ķֿ飬�������ʧ��
 */
void encodeFile(const Context* context, int n, int m);
void decodeFile(const Context* context, int n, int m);

#endif
