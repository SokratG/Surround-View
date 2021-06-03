#include "util.cuh"

__host__ inline int divUp(int a, int b){
        return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void copySplitView_kernel(const cv::cuda::PtrStep<uchar*> src, cv::cuda::PtrStep<uchar*> left_dst, cv::cuda::PtrStep<uchar*> right_dst,
                                     const int half_width, const uint width, const uint height)
{



}


void copySplitView(const cv::cuda::PtrStep<uchar*> src, cv::cuda::PtrStep<uchar*> left_dst, cv::cuda::PtrStep<uchar*> right_dst,
                   const int half_width, const uint width, const uint height)
{



}



