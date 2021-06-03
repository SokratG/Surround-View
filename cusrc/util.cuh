#pragma once
#include <cuda_runtime.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/types.hpp>

void copySplitView(const cv::cuda::PtrStep<uchar*> src, cv::cuda::PtrStep<uchar*> left_dst, cv::cuda::PtrStep<uchar*> right_dst,
                   const int half_width, const uint width, const uint height);

