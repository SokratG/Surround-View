#pragma once
#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef unsigned int uint;

void Gray_gpu(uchar* imgarr, uint width, uint height);
void ImgDiff_gpu(float* imgin1, float* imgin2, float* imgout, uint width, uint height, float s);
void detectKeyPoints_gpu(float* imgleft, float* imgright, float* imgres, float* keyPoints, uint width, uint height, uint chan, int r, float threshold);
 
