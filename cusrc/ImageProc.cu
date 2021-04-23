#include "ImageProc.cuh"



__global__ void Gray_gpu_kernel(uchar* imgarr, uint w, uint h)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if (x >= w || h <= y)
		return;
	
	uint idx = ((y * w) + x) * 3;
	uint tempR = imgarr[idx++];
	uint tempG = imgarr[idx++];
	uint tempB = imgarr[idx++];
	
	tempR = tempG = tempB = (tempR+tempG+tempB) / 3;
	imgarr[--idx] = tempB;
	imgarr[--idx] = tempG;
	imgarr[--idx] = tempR;
}

__global__ void ImgDiff_gpu_kernel(float* imgin1, float* imgin2, float* imgout, uint w, uint h, float s)
{
	
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= w || h <= y)
		return;
	
	uint idx = ((y * w) + x) * 3;
	
	imgout[idx] = abs((imgin2[idx] - imgin1[idx])/(1-s));
	imgout[idx+1] = abs((imgin2[idx+1] - imgin1[idx+1])/(1-s));
	imgout[idx+2] = abs((imgin2[idx+2] - imgin1[idx+2])/(1-s));
}

__global__ void detectKeyPoints_gpu_kernel(float* imgleft, float* imgright, float* imgres, float* keyPoints, uint w, uint h, uint chan, int r, float threshold)
{

	extern __shared__ float  picBlock[];
	
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if(i >= w || j >= h)
		return;
	
	if(i < r || i >= (w-r) || j < r || j >= (h-r)){
		return;
	}
	
	int idx = (j * w + i) * chan;
			
	float dx = 0.0;
	float dy = 0.0;
	int patchCount = 0;
	
	//Calculate the x,y gradient by considering neighbouring pixels.
	for(int k = 0; k < 3; k++){
		float * curr;
		switch(k){
			case 0: curr = imgleft; break;
			case 1: curr = imgright; break;
			case 2: curr = imgres; break;
		}
		for(int y = j-r+1; y <= j+r; y++){
			for(int x = i-r+1; x <= i+r ; x++){
				int kernIdx =  (y*w+x)*chan;
				dy = (y>j) ? dy + curr[kernIdx]: dy - curr[kernIdx];
				dx = (x>i) ? dx + curr[kernIdx]: dx - curr[kernIdx];
				patchCount++;
			}
		}
	}
	patchCount /= 2;
	//Average the gradients.
	dx /= patchCount;
	dy /= patchCount;
	
	float ddx = 0;
	float ddy = 0;
	patchCount = 0;
	//Calculate the x,y second gradient by considering neighbouring pixels.
	for(int k = 0; k < 3; k++){
		float * curr;
		switch(k){
			case 0: curr = imgleft; break;
			case 1: curr = imgright; break;
			case 2: curr = imgres; break;
		}
		for(int y = j-r+1; y <= j+r; y++){
			for(int x = i-r+1; x <= i+r ; x++){
				int kernIdx =  (y*w+x)*chan;
				ddy = abs(y-j)>r ? dy + curr[kernIdx]: 2*(dy - curr[kernIdx]);
				ddx = abs(x-i)>r ? dx + curr[kernIdx]: 2*(dx - curr[kernIdx]);
				patchCount++;
			}
		}
	}

	patchCount /= 2;
	//Average the gradients.
	ddx /= patchCount;
	ddy /= patchCount;
		
	
	float D = imgright[idx];
	
	__syncthreads();
	//Early termination. These points are unlikely to be keypoints
	//Anyway.
	if(D < threshold ){
		keyPoints[idx+0] += 0.0f;
		keyPoints[idx+1] += 0.0f;
		keyPoints[idx+2] += 0.0f;			
		return;
	}
	
	/*
	* All the subblocks are now in shared memory. Now we blurr the image.
	*/

	
	//Key point localization.
	//Eliminate points along the edges.
	float a = dx * dx;
	float b = 2 * dx * dy;
	float c = dy * dy;
	
	float elipDet = (b * b + (a - c) * (a - c));

	float l1 = 0.5 * (a + c + sqrt(elipDet));
	float l2 = 0.5 * (a + c - sqrt(elipDet));
	
	float R = (l1 * l2 - 1e-8*(l1 + l2) * (l1 + l2));
	
	float T1 = 1e-3;
	if( abs(R) < T1 ){
		//Eliminate points along edges							
		keyPoints[idx+0] += 0;	
		keyPoints[idx+1] += 0;	
		keyPoints[idx+2] += 0;	
		return;
	}else if( R < T1){
		//Eliminate points along edges							
		keyPoints[idx+0] += 0;	
		keyPoints[idx+1] += 0;	
		keyPoints[idx+2] += 0;	
		return;
	}else{
		//Corners
		//keyPoints[idx+2] += 1;	//R
		;
	}
	//imgleft
	//imgright
	//imgres
	patchCount = 0;
	float ds_f =  0;	//Forwards
	float ds_b =  0;	//Backwards	
	for(int y = j-r+1; y <= j+r; y++){
		for(int x = i-r+1; x <= i+r ; x++){
			int kernIdx =  (y*w+x)*chan;
			ds_f +=  (imgres[kernIdx] -  imgright[kernIdx]);	//Forwards
			ds_b +=  (imgright[kernIdx] -  imgleft[kernIdx]);	//Backwards
			patchCount++;
		}
	}
	
	patchCount/=2;
	//First Derivative.
	ds_f /=  patchCount;	//Forwards
	ds_b /=  patchCount;	//Backwards
	
	//Average them.
	float ds = (ds_f + ds_b)/2;
	float dxVec[3] = {-dx,-dy,-ds};

	
	//Second Derivatives
	float dds = ds_f - ds_b;
	
	//Second derivative matrix

	float DDMat[3][3] ={	{ddx,	dx*dy,	dx*ds},
				{dy*dx,	ddy,	dy*ds},
				{ds*dx,	ds*dy,	dds}  };

	//Now get its inverse.
	float det =(DDMat[0][0]*DDMat[1][1]*DDMat[2][2] +
				DDMat[0][1]*DDMat[1][2]*DDMat[2][0] +
				DDMat[0][2]*DDMat[1][0]*DDMat[2][1]	) 
				-
			   (DDMat[0][2]*DDMat[1][1]*DDMat[2][0] +
				DDMat[0][1]*DDMat[1][0]*DDMat[2][2] +
				DDMat[0][0]*DDMat[1][2]*DDMat[2][1]	);

	
	if(det != 0){
		
		//Adjugate matrix. Matrix of coffactors.
		float CC_00 = DDMat[1][1]*DDMat[2][2] - DDMat[1][2]*DDMat[2][1];
		float CC_01 = DDMat[0][2]*DDMat[2][1] - DDMat[0][1]*DDMat[2][2];
		float CC_02 = DDMat[0][1]*DDMat[1][2] - DDMat[0][2]*DDMat[1][1];

		float CC_10 = DDMat[1][2]*DDMat[2][0] - DDMat[1][0]*DDMat[2][2];
		float CC_11 = DDMat[0][0]*DDMat[2][2] - DDMat[0][2]*DDMat[2][0];
		float CC_12 = DDMat[0][2]*DDMat[1][0] - DDMat[0][0]*DDMat[1][2];

		float CC_20 = DDMat[1][0]*DDMat[2][1] - DDMat[1][1]*DDMat[2][0];
		float CC_21 = DDMat[0][1]*DDMat[2][0] - DDMat[0][0]*DDMat[2][1];
		float CC_22 = DDMat[0][0]*DDMat[1][1] - DDMat[0][1]*DDMat[1][0];
	
		float CCMat[3][3] ={{CC_00,	CC_01,	CC_02},
							{CC_10,	CC_11,	CC_12},
							{CC_20,	CC_21,	CC_22}};
	
		//Inverse matrix.
		CCMat[0][0] /= det;	CCMat[0][1] /= det;	CCMat[0][2] /= det;
		CCMat[1][0] /= det;	CCMat[1][1] /= det;	CCMat[1][2] /= det;
		CCMat[2][0] /= det;	CCMat[2][1] /= det;	CCMat[2][2] /= det;

		//Aproximation factors
		float XBarVec[3]=	{CCMat[0][0]*dxVec[0]+CCMat[0][1]*dxVec[1]+CCMat[0][2]*dxVec[2],
							CCMat[1][0]*dxVec[0]+CCMat[1][1]*dxVec[1]+CCMat[1][2]*dxVec[2],
							CCMat[2][0]*dxVec[0]+CCMat[2][1]*dxVec[1]+CCMat[2][2]*dxVec[2] };
		
		//Remove low contrast extrema.
		float xbarThr = 0.5f;
		if( ( abs( XBarVec[0] ) > xbarThr || abs( XBarVec[1] ) > xbarThr || abs( XBarVec[2] ) > xbarThr ) ){
			keyPoints[idx+0] += 0;	
			keyPoints[idx+1] += 0;	
			keyPoints[idx+2] += 0;	
			return;
		}else{
			D += (XBarVec[0]*dxVec[0]+ XBarVec[1]*dxVec[1] + XBarVec[2]*dxVec[2])/2.0f;
			keyPoints[idx+1] += D;
			return;
		}
	}	
	
	keyPoints[idx+0] += 0;	
	keyPoints[idx+1] += 0;	
	keyPoints[idx+2] += 0;		

}

void Gray_gpu(uchar* imgarr, uint width, uint height)
{
	if (imgarr == NULL)
		return; 
	uint blockSize = 1024;
	uint numBlocks = (width / 2 + blockSize - 1) / blockSize;
	Gray_gpu_kernel <<<numBlocks, blockSize >>>(imgarr, width, height);
}

void ImgDiff_gpu(float* imgin1, float* imgin2, float* imgout, uint width, uint height, float s)
{
	if (imgin1 == NULL || imgin2 == NULL || imgout == NULL)
		return; 

	uint blockSize = 1024;
	uint numBlocks = (width / 2 + blockSize - 1) / blockSize;
	ImgDiff_gpu_kernel <<<numBlocks, blockSize >>>(imgin1, imgin2, imgout, width, height, s);

}


void detectKeyPoints_gpu(float* imgleft, float* imgright, float* imgres, float* keyPoints, uint width, uint height, uint chan, int r, float threshold)
{



}











