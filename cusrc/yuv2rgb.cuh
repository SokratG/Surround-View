#pragma once
#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef unsigned int uint;

void gpuConvertUYVY2RGB(uchar* src, uchar* dst, uint width, uint height);
void gpuConvertUYVY2RGB_async(uchar* src, uchar* dst, uint width, uint height, cudaStream_t stream);
void gpuConvertUYVY2RGB_opt(uchar* src, uchar* d_src, uchar* dst, uint width, uint height, cudaStream_t stream);

