#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef unsigned int uint;


void gpuConvertUYVY2RGB_async(uchar* src, uchar* d_src, uchar* dst, uint width, uint height, cudaStream_t stream);

static bool cudaHandleError(cudaError_t err, const char* file, int line)
{
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        return false;
    }
    return true;
}

#define CUHANDLE_ERROR(err) (cudaHandleError(err, __FILE__, __LINE__))
