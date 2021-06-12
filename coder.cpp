#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <tmmintrin.h>
#include "coder.h"

#define GET_BLOCK_SIZE(_SIZE,_NUM) ((_SIZE+_NUM-(_SIZE%_NUM))/_NUM)

const int BUF_LEN = 256;
const int NAME_LEN = 32;
const char* FORMAT = "%s_%d.dat";
matrix_t g_matrix;

clock_t start, finish;

__m128i ax ;
__m128i bx ;
__m128i cx ;
__m128i dx ;

extern UInt8_t parallel_low_table[256][16];
extern UInt8_t parallel_high_table[256][16];


void matrixDisPlay(matrix_t* A);

/************************************************************************
 * initMatrix - 初始化生成矩阵g_matrix
 * @n : 矩阵的行数
 * @m : 矩阵的列数
 ************************************************************************/
void initMatrix(int n, int m) {
	int i, j;
	g_matrix.m_row = m;
	g_matrix.m_col = n;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			g_matrix.m_data[i][j] = galoisPow(j + 1, i);
		}
	}
}

/************************************************************************
 * xorMulArr - 将src数组中元素乘以系数num然后加到dest中去
 * @num : src数据要乘的系数
 * @src : 一组数据
 * @dest : 一组数据
 * @size : 要加到dest数组去的元素的个数
 ************************************************************************/
void xorMulArr(UInt8_t num, UInt8_t* src, UInt8_t* dest, int size) {
	int i;
	if (!num) {
		return;
	}
	for (i = 0; i < size; ++i) {
		(*dest++) ^= galoisMul(num, *src++);
	}
}

/************************************************************************
 * dealbuffer - 处理从文件中读入内存的一块数据
 * @context : 原始文件的上下文
 * @c : 在编码过程中使用到的内存资源，m个冗余块对应分配m个数组
 * @buf : 存放着从原始文件中读入到内存里的数据
 * @m : 要编码的冗余块的个数
 * @buflen : buf的长度
 * @offset : 当前buf中的数据在文件中的偏移位置
 * @blen: 每一个块的长度
 ************************************************************************/
void dealbuffer(const Context* context, UInt8_t** c, UInt8_t* buf,
	int m, int buflen, int offset, int blen) {
	int i, cursize, pos = 0;
	char name[NAME_LEN];
	FILE* fp;
	UInt8_t coef;

	int j,k,l=0;

	// 计算当前位置处于哪个块，及在块中的偏移量
	int bnum = offset / blen;
	int boff = offset % blen;

	while (pos < buflen) {
		// 获取一次要编码到当前block中的数据长度
		// 要么buffer结束， 要么当前块结束
		if (blen - boff > buflen - pos) {
			cursize = buflen - pos;
		}
		else {
			cursize = blen - boff;
		}


/*
		for (i = 0; i < m; ++i) {
			k=0;
			while(cursize-16*k>=0)
			{
				coef = g_matrix.m_data[i][bnum];
				for(j=0;j<16;j++)
				{
					ax.m128i_i8[j]=parallel_low_table[coef][j];
				}
				
				

				for(l=0;l<((cursize-16*k-16)>16?16:cursize-16*k-16);l++)
				{
					bx.m128i_i8[l]=(buf+pos)[l] & 0x0f;
				}
				cx=_mm_shuffle_epi8(ax,bx);

				for(j=0;j<16;j++)
				{
					ax.m128i_i8[j]=parallel_high_table[coef][j];
				}
				
				for(l=0;l<((cursize-16*k-16)>16?16:cursize-16*k-16);l++)
				{
					bx.m128i_i8[l]=((buf+pos)[l] >>4 ) & 0xf0;
				}
				dx=_mm_shuffle_epi8(ax,bx);


				for(l=0;l<((cursize-16*k-16)>16?16:cursize-16*k-16);l++)
				{
					(c[i] + boff)[l]   ^=     cx.m128i_u8[l]^dx.m128i_u8[l];
				}


				k++;
			}
			
		}
		*/
		// 将buf中cursize的数据同个伽罗华计算到对应的块中
	
		for (i = 0; i < m; ++i) {
			coef = g_matrix.m_data[i][bnum];
			xorMulArr(coef, buf + pos, c[i] + boff, cursize);
		}
		
	
	
		// 将pos到pos+cursize的数据追加到bnum文件中   这是文件分片的过程
		sprintf_s(name, FORMAT, context->fname, bnum);
		fp = fopen(name, "ab+");
		fwrite(buf + pos, 1, cursize, fp);
		fclose(fp);

		// pos bnum boff 相应改变继续处理后续数据
		pos += cursize;
		bnum++;
		boff = 0;
	}
}

/************************************************************************
 * clearBlock - 清除和一个文件相关的所有块
 * @context : 原始文件的上下文
 * @num : 文件相关的分割块和冗余块数目总和
 ************************************************************************/
void clearBlock(const Context* context, int num) {
	int i;
	char name[NAME_LEN];
	for (i = 0; i < num; ++i) {
		sprintf_s(name, FORMAT, context->fname, i);
		remove(name);
	}
}

/************************************************************************
 * encodeFile - 编码一个文件
 * @context : 原始文件的上下文
 * @n : 原始文件要分割的块数目
 * @m : 原始文件要编码的冗余块数目
 ************************************************************************/
void encodeFile(const Context* context, int n, int m) {
	int blen, i , j , k ,  num, offset;
	UInt8_t buffer[BUF_LEN];
	char name[NAME_LEN];
	FILE* pm;
	FILE* pf = fopen(context->fname, "rb");
	if (!pf) {
		return;
	}
	initMatrix(n, m);
	blen = GET_BLOCK_SIZE(context->flen, n);     //数据块的字节数

	// 在编码过程中使用到的内存资源，m个冗余块对应分配m个数组
	UInt8_t** c = (UInt8_t**)malloc(sizeof(int*) * m);
	for (i = 0; i < m; ++i) {
		c[i] = (UInt8_t*)malloc(sizeof(UInt8_t) * blen);
		memset(c[i], 0, sizeof(UInt8_t) * blen);
	}

	//为什么编码速度如此之慢34KB/s，这里做一个实验：先把文件数据读到数组里，从这个数组开始编码，看看时间会如何变换
	UInt8_t** data_array = (UInt8_t**)malloc(sizeof(int*) * n);
	for (i = 0; i < n; ++i) {
		data_array[i] = (UInt8_t*)malloc(sizeof(UInt8_t) * blen);
		memset(data_array[i], 0, sizeof(UInt8_t) * blen);
	}

	for (i = 0; i < n; ++i) {
		fread(data_array[i], sizeof(UInt8_t), blen, pf);
	}

	// 每次编码前清除原有的块
	clearBlock(context, m + n);

	printf("读取文件阶段一完成\n");
	start = clock();
	printf("编码开始的时间为%f ms\n", double(start));
	offset = 0;
	for (i = 0; i < n; ++i) {                  //原始块
		for (j = 0; j < m; ++j) {			   //冗余块
			for (k = 0; k < blen; ++k) {	   //

				xorMulArr(g_matrix.m_data[j][i], data_array[i] + k, c[j] + k, 1);
			}
			
		}
		
	}
	
	finish = clock();

	printf("编码结束的时间为%f ms\n", double(finish));
	printf("编码时间为%f ms\n", double(finish - start));
	printf("编码速度为%f KB/s\n", double(30301000 / (finish - start)));


	// 从原始文件里读数据，并做分割和编码处理
	// 每次读入到buffer中的数据，由dealbuffer来处理
	//offset = 0;
	//while (num = fread(buffer, sizeof(UInt8_t), BUF_LEN, pf)) {
	//	dealbuffer(context, c, buffer, m, num, offset, blen);
	//	offset += num * sizeof(UInt8_t);
	//}

	
	

	// 将数据分片写入磁盘
	for (i = 0; i < n; ++i) {
		sprintf_s(name, FORMAT, context->fname,  i);
		pm = fopen(name, "wb");
		fwrite(data_array[i], sizeof(UInt8_t), blen, pm);
		fclose(pm);
	}

	// 将编码得到的冗余块写入磁盘
	for (i = 0; i < m; ++i) {
		sprintf_s(name, FORMAT, context->fname, n + i);
		pm = fopen(name, "wb");
		fwrite(c[i], sizeof(UInt8_t), blen, pm);
		fclose(pm);
	}

	// 所有编码工作完毕，释放资源
	for (i = 0; i < m; ++i) {
		free(c[i]);
	}
	free(c);

	for (i = 0; i < n; ++i) {
		free(data_array[i]);
	}
	free(data_array);
	fclose(pf);
}

/************************************************************************
 * encodeFile - 编码一个文件
 * @context : 原始文件的上下文
 * @n : 原始文件要分割的块数目
 * @m : 原始文件要编码的冗余块数目
 ************************************************************************/
void fillerase(UInt8_t** e, char* fname, int begin, int end, int n, int m) {
	int i, j, pos, num, coef;
	char name[NAME_LEN];
	UInt8_t buf[BUF_LEN];
	FILE* fp;
	for (i = begin; i < end; ++i) {
		sprintf_s(name, FORMAT, fname, i);
		fp = fopen(name, "rb+");
		if (!fp) {
			continue;
		}
		pos = 0;
		while (num = fread(buf, 1, BUF_LEN, fp)) {
			for (j = 0; j < m; ++j) {
				if (i >= n && j != i - n) {
					continue;
				}
				coef = (i >= n) ? 1 : g_matrix.m_data[j][i];
				xorMulArr(coef, buf, e[j] + pos, num);
			}
			pos += num;
		}
	}
}

/************************************************************************
 * getIndexInfo - 获取所有已丢失的原始块编号，和未丢失的冗余块编号
 * @fname : 原始文件名
 * @ib : 存放所有已丢失的原始块的编号的数组
 * @ic : 存放所有未丢失的冗余块的编号的数组
 * @lenb : 数组ib的长度
 * @lenc : 暑促ic的长度
 * @m : 编码方案中冗余块的个数
 * @n : 编码方案中原始块的个数
 ************************************************************************/
void getIndexInfo(char* fname, int* ib, int* ic, int* lenb,
	int* lenc, int m, int n) {
	int i;
	FILE* fp;
	char name[NAME_LEN];
	for (i = 0; i < n + m; ++i) {
		sprintf_s(name, FORMAT, fname, i);
		fp = fopen(name, "rb+");
		// lenb 丢失的普通块个数
		if (!fp) {
			if (i < n) {
				ib[(*lenb)++] = i;
			}
			continue;
		}
		// lenc 存在的编码块个数
		if (i >= n) {
			ic[(*lenc)++] = i;
		}
		fclose(fp);
	}
}

/************************************************************************
 * createMatrix - 根据已丢失的原始块编号和未丢失冗余块编号构造系数矩阵
 * @ib : 存放所有已丢失的原始块的编号的数组
 * @ic : 存放所有未丢失的冗余块的编号的数组
 * @lenb : 数组ib的长度
 * @lenc : 暑促ic的长度
 * @n : 编码方案中原始块的个数
 ************************************************************************/
matrix_t createMatrix(int* ib, int* ic, int lenb, int lenc, int n) {
	int i, j;
	matrix_t A;
	A.m_row = lenb;
	A.m_col = lenb;
	for (j = 0; j < lenb; ++j) {
		for (i = 0; i < lenb; ++i) {
			A.m_data[j][i] = g_matrix.m_data[ic[j] - n][ib[i]];
		}
	}
	return A;
}

/************************************************************************
 * createMatrix - 根据系数矩阵的逆矩阵A和era，来解码求解已丢失的块
 * @A : 系数矩阵的逆矩阵
 * @era : 每个冗余块对应的已经消项的数组
 * @fname : 原始文件名
 * @ic : 存放所有未丢失的冗余块的编号的数组
 * @ib : 存放所有已丢失的原始块的编号的数组
 * @lenb : 数组ib的长度
 * @lenc : 暑促ic的长度
 * @blen : 每个block的长度
 * @n : 编码方案中原始块的个数
 ************************************************************************/
void buildBlock(matrix_t* A, UInt8_t** era, char* fname, int* ic,
	int* ib, int lenb, int lenc, int blen, int n) {
	int i, j, coef;
	char name[NAME_LEN];
	FILE* fp;
	UInt8_t* buf = (UInt8_t*)malloc(sizeof(UInt8_t) * blen);
	for (i = 0; i < lenb; ++i) {
		memset(buf, 0, sizeof(UInt8_t) * blen);
		sprintf_s(name, FORMAT, fname, ib[i]);
		for (j = 0; j < lenc; ++j) {
			coef = A->m_data[i][j];
			xorMulArr(coef, era[ic[j] - n], buf, blen);
		}
		fp = fopen(name, "wb");
		fwrite(buf, 1, blen, fp);
		fclose(fp);
	}
}

/************************************************************************
 * buildFile - 将已经解码出的所有块组合成原始文件
 * @context : 原始文件相关信息
 * @n : 编码方案中要分割出的原始块的个数
 ************************************************************************/
void buildFile(const Context* context, int n) {
	int i, num, cursize, curlen;
	char name[NAME_LEN];
	char buf[BUF_LEN];
	FILE* fp = fopen(context->fname, "wb");
	FILE* fb;
	curlen = 0;
	for (i = 0; i < n; ++i) {
		sprintf_s(name, FORMAT, context->fname, i);
		fb = fopen(name, "rb+");
		if (!fb) {
			printf("decode failed!\n");
			fclose(fp);
			return;
		}
		cursize = 0;
		while (num = fread(buf, 1, BUF_LEN, fb)) {
			if (curlen + num > context->flen) {
				cursize = context->flen - curlen;
			}
			else {
				cursize = num;
			}
			fwrite(buf, 1, cursize, fp);
		}
		fclose(fb);
	}
	fclose(fp);
}
/************************************************************************
 * createMatrix - 如果有冗余块丢失，再编码出丢失的冗余块
 * @context : 原始文件上下文
 * @era : 每个冗余块对应的已经消项的数组
 * @ib : 存放所有已丢失的原始块的编号的数组
 * @ic : 存放所有未丢失的冗余块的编号的数组
 * @lenb : 数组ib的长度
 * @lenc : 数组ic的长度
 * @m : 编码方案中冗余块的个数
 * @n : 编码方案中原始块的个数
 * @blen : 每个block的长度
 ************************************************************************/
void buildBackups(const Context* context, UInt8_t** era, int* ib,
	int* ic, int lenb, int lenc, int m, int n, int blen) {
	int i, j, flag, num, pos;
	char name[NAME_LEN];
	UInt8_t buf[BUF_LEN];
	FILE* fc, * fb;
	UInt8_t coef;

	for (i = 0; i < m; ++i) {
		flag = 0;
		// 判断第i个冗余块是否已经存在
		for (j = 0; j < lenc; ++j) {
			if (i == ic[j] - n) {
				flag = 1;
			}
		}
		if (flag) {
			continue;
		}
		// 如果不存在，则将丢失的普通块都编码进对应的era处
		for (j = 0; j < lenb; ++j) {
			coef = g_matrix.m_data[i][ib[j]];
			sprintf_s(name, FORMAT, context->fname, ib[j]);
			fb = fopen(name, "rb");
			pos = 0;
			while (num = fread(buf, sizeof(UInt8_t), BUF_LEN, fb)) {
				xorMulArr(coef, buf, era[i] + pos, num);
				pos += num;
			}
			fclose(fb);
		}
		sprintf_s(name, FORMAT, context->fname, n + i);
		fc = fopen(name, "wb+");
		fwrite(era[i], sizeof(UInt8_t), blen, fc);
		fclose(fc);
	}
}

/************************************************************************
 * decodeFile - 解码指定上下文的文件
 * @context : 原始文件上下文
 * @m : 编码方案中冗余块的个数
 * @n : 编码方案中原始块的个数
 ************************************************************************/
void decodeFile(const Context* context, int n, int m) {
	int blen, i, lenb, lenc;
	matrix_t A, B;

	// ib数组存放丢失的普通块编号
	// ic数组存放存在的冗余块编号
	int* ib = (int*)malloc(sizeof(int) * (n + m));
	int* ic = (int*)malloc(sizeof(int) * (n + m));
	UInt8_t** era;
	blen = GET_BLOCK_SIZE(context->flen, n);
	lenb = lenc = 0;

	// 初始化生成矩阵
	initMatrix(n, m);

	// 获取丢失的普通块编号，和存在的冗余块编号
	getIndexInfo(context->fname, ib, ic, &lenb, &lenc, m, n);

	// 丢失的普通块个数大于存在的编码块个数，无法编码
	if (lenb > lenc) {
		printf("can not decode!");
		return;
	}

	// 分配消项过程中使用到的内存，并通过fillerase对冗余块进行消项
	era = (UInt8_t**)malloc(sizeof(UInt8_t*) * m);
	for (i = 0; i < m; ++i) {
		era[i] = (UInt8_t*)malloc(sizeof(UInt8_t) * blen);
		memset(era[i], 0, sizeof(UInt8_t) * blen);
	}
	fillerase(era, context->fname, 0, n, n, m);
	fillerase(era, context->fname, n, n + m, n, m);

	// 通过丢失的普通块编号和存在的冗余块编号，来确定小矩阵A
	// 通过高斯约旦消去法求出矩阵A的逆矩阵
	A = createMatrix(ib, ic, lenb, lenc, n);
	B = matrixGauss(&A);

	// 逆矩阵和era数组相乘，可计算得到丢失的普通快
	buildBlock(&B, era, context->fname, ic, ib, lenb, lenc, blen, n);

	// 构造原始文件
	buildFile(context, n);

	// 如果有被删除的冗余块，则需要构造冗余块
	if (lenc != m) {
		buildBackups(context, era, ib, ic, lenb, lenc, m, n, blen);
	}
	// 解码完毕，释放资源
	for (i = 0; i < m; ++i) {
		free(era[i]);
	}
	free(era);
	free(ib);
	free(ic);
}

int main(void) {
	int i, j;
	char command;
	char name1[NAME_LEN];
	char name2[NAME_LEN];
	// 初始化伽罗华域
	galoisEightBitInit();
//	galois8_parallel_tableInit();

	

	// 构造要编解码的文件的上下文信息
	Context context;
	context.fname = "z_back.mp4";
	FILE* pf = fopen(context.fname, "r+b");
	if (pf==NULL) {
		return 1;
	}
	printf("开始读取文件\n");
	fseek(pf, 0, SEEK_END);        //文件的字节数
	context.flen = ftell(pf);
	if (fclose(pf)) {
		printf("fclose error!\n");
	}
	printf("读取文件阶段一完成\n");
	 //编码
	encodeFile(&context, 8, 2);
	printf("编码结束了\n");
	
	printf("input the delete file's index : ");
	fflush(stdin);
	scanf("%d", &i);
	scanf("%d", &j);
	sprintf_s(name1, FORMAT, context.fname, i);
	sprintf_s(name2, FORMAT, context.fname, j);
	printf("delete file as follow!\n");
	remove(context.fname);
	printf("  %s\n", name1);
	remove(name1);
	printf("  %s\n", name2);
	remove(name2);
	printf("please check!\n");
	printf("Do you want to decode file (y/n)? : ");
	getchar();
	fflush(stdin);
	scanf("%c", &command);
	if (command == 'y') {
		// 解码
		decodeFile(&context, 8, 2);
	}





	return 0;
}
