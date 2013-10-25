/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer
#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;
cudaArray *d_histogramArray = 0; //added
cudaArray *d_codebookArray = 0;
cudaArray *d_templatesArray = 0;
cudaArray *d_codebookTemplateIdArray = 0;
cudaArray *d_errorsbookArray = 0;

// flexible block size
cudaArray *d_codebookSpanLowArray = 0;
cudaArray *d_codebookSpanHighArray = 0;
cudaArray *d_flexibleCodebookArray = 0;
cudaArray *d_flexibleErrorsbookArray = 0;
cudaArray *d_simpleSpanLowArray = 0;
cudaArray *d_simpleSpanHighArray = 0;
cudaArray *d_simpleCountArray = 0;
cudaArray *d_simpleHistogramArray = 0;
cudaArray *d_flexibleTemplatesArray = 0;
cudaArray *d_flexTexArray = 0;

cudaArray *d_originalQueryArray = 0;
cudaArray *d_fractalQueryArray = 0;

//typedef unsigned char VolumeType;
typedef unsigned int VolumeType; //NOTE: here we will store index number of histogram data into the volume, index number is unsigned int
//typedef unsigned short VolumeType;
typedef float HistogramType;
typedef int4 CodebookType;
typedef float TemplatesType; // NOTE: CUDA kernel doesn't support double (instead, it only support double1, double2), or it will report error when creating channel descriptor
typedef int CodebookTemplateIdType;
typedef float2 ErrorsbookType;

typedef int4 SpanType;
typedef int4 FlexibleCodebookType;
typedef float2 FlexibleErrorsbookType;
typedef int SimpleCountType;
typedef float2 SimpleHistogramType;
typedef float4 FlexTexType;

//texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;									// 3D texture
texture<VolumeType, cudaTextureType3D, cudaReadModeElementType> tex; // NOTE: here we don't want the index to be normalized, so we use elementtype
texture<float4, 1, cudaReadModeElementType> transferTex; // 1D transfer function texture
texture<HistogramType, cudaTextureType2DLayered, cudaReadModeElementType> histogramTex; // the texture that store histogram
texture<CodebookType, cudaTextureType3D, cudaReadModeElementType> codebookTex; // 3D texture for codebook
texture<TemplatesType, cudaTextureType2DLayered, cudaReadModeElementType> templatesTex; // 2D layered texture for templates
texture<CodebookTemplateIdType, cudaTextureType3D, cudaReadModeElementType> codebookTemplateIdTex;
texture<ErrorsbookType, cudaTextureType2DLayered, cudaReadModeElementType> errorsbookTex;

// flexible block size
texture<SpanType, cudaTextureType3D, cudaReadModeElementType> codebookSpanLowTex;
texture<SpanType, cudaTextureType3D, cudaReadModeElementType> codebookSpanHighTex;
texture<FlexibleCodebookType, cudaTextureType3D, cudaReadModeElementType> flexibleCodebookTex;
texture<FlexibleErrorsbookType, cudaTextureType2DLayered,
		cudaReadModeElementType> flexibleErrorsbookTex;
texture<SpanType, cudaTextureType3D, cudaReadModeElementType> simpleSpanLowTex;
texture<SpanType, cudaTextureType3D, cudaReadModeElementType> simpleSpanHighTex;
texture<SimpleCountType, cudaTextureType3D, cudaReadModeElementType> simpleCountTex;
texture<SimpleHistogramType, cudaTextureType2DLayered, cudaReadModeElementType> simpleHistogramTex;
texture<TemplatesType, cudaTextureType2DLayered, cudaReadModeElementType> flexibleTemplatesTex;

texture<FlexTexType, cudaTextureType3D, cudaReadModeElementType> flexBlockTex;
cudaTextureObject_t flexBlockTexObj;
cudaSurfaceObject_t flexBlockSurfObj = 0;

typedef float4 QueryType;
texture<QueryType, cudaTextureType3D, cudaReadModeElementType> originalQueryTex; // NOTE: here we don't want the index to be normalized, so we use elementtype
texture<QueryType, cudaTextureType3D, cudaReadModeElementType> fractalQueryTex;

const int nBlocks = (int) 50 * 50 * 10;
const int nBins = 32;

const int nMaxBlockDim = 500; // this means we allow at lease 2*2*2 size block
const int nLgTwo = 6;

const int nDimension = 64;
const int flexNBin = 64;
const int nTemplate = 469; // TODO: WARNING! this number needs to be changed!
const cudaExtent flexibleVolumeSize = make_cudaExtent(64, 64, 32); // NOTE: here we just hard code the number of fractal histograms or simple histograms, each is 131072
const cudaExtent flexibleHistogramSize = make_cudaExtent(64, 2048, 64); // for simpleHistogram, 64*64*32 blocks, 64 bins, width is the bin number
const cudaExtent flexibleTemplatesSize = make_cudaExtent(64, 469, 1); // NOTE: here we hard code the size
const cudaExtent flexTexSize = make_cudaExtent(nMaxBlockDim, nMaxBlockDim,
		nMaxBlockDim);
const dim3 rawVolumeDim(64, 64, 64);
const cudaExtent originalVolumeSize = make_cudaExtent(50, 50, 10);

typedef struct {
	float4 m[3];
} float3x4;

typedef struct {
	int4 low;
	int4 high;
} Span;

__constant__ float3x4 c_invViewMatrix; // inverse view matrix

struct Ray
{
	float3 o; // origin
	float3 d; // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

// __device__ Span* flexBlock; // span of each block
// NOTE: here we hard code the flexBlock
__device__ Span flexBlock[4096];	// 16*16*16

__device__ int nFlexBlock; // total number of blocks in this volume
__device__ int nFlexBlockX; // number of blocks in x dimension
__device__ int nFlexBlockY;
__device__ int nFlexBlockZ;

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear,
		float *tfar) {
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

__device__
int d_spanSize(Span s) {
	int sizeX = s.high.x - s.low.x + 1;
	int sizeY = s.high.y - s.low.y + 1;
	int sizeZ = s.high.z - s.low.z + 1;
	return sizeX * sizeY * sizeZ;
}


// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v) {
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16)
			| (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__device__
float *fractalDecoding(float original[nBins], int flip, int shift) {
	float decoded[nBins];
	//float *decoded;
	//decoded = new float[nBins];
	if (flip == 0) {
		for (int i = 0; i < nBins; i++) {
			int m = i + shift;
			if (m >= nBins) {
				m = m - nBins;
			}
			decoded[m] = original[i];
		}
	} else {
		float temp[nBins];
		for (int i = 0; i < nBins; i++) {
			temp[i] = original[nBins - 1 - i];
		}
		for (int i = 0; i < nBins; i++) {
			int m = i + shift;
			if (m >= nBins) {
				m = m - nBins;
			}
			decoded[m] = temp[i];
		}
	}
	return decoded;
}

__device__
float *flexibleFractalDecoding(float original[flexNBin], int flip, int shift) {
	float decoded[flexNBin];
	//float *decoded;
	//decoded = new float[nBins];
	if (flip == 0) {
		for (int i = 0; i < flexNBin; i++) {
			int m = i + shift;
			if (m >= flexNBin) {
				m = m - flexNBin;
			}
			decoded[m] = original[i];
		}
	} else {
		float temp[flexNBin];
		for (int i = 0; i < flexNBin; i++) {
			temp[i] = original[flexNBin - 1 - i];
		}
		for (int i = 0; i < flexNBin; i++) {
			int m = i + shift;
			if (m >= flexNBin) {
				m = m - flexNBin;
			}
			decoded[m] = temp[i];
		}
	}
	return decoded;
}

__device__
bool inInterpolation(float3 pos, float3 interPosBot, float3 interPosTop, int x,
		int y) {
	pos.x = pos.x * 0.5f + 0.5f;
	pos.y = pos.y * 0.5f + 0.5f;
	pos.z = pos.z * 0.5f + 0.5f;
	//if (x == 200 && y == 200) {
	//	printf("pos\t\t %f %f %f\n", pos.x, pos.y, pos.z);
	//	printf("InterPosBot\t %f %f %f\n", interPosBot.x, interPosBot.y, interPosBot.z);
	//	printf("interPosTop\t %f %f %f\n", interPosTop.x, interPosTop.y, interPosTop.z);
	//}
	if (pos.x < interPosBot.x || pos.y < interPosBot.y || pos.z < interPosBot.z
			|| pos.x > interPosTop.x || pos.y > interPosTop.y
			|| pos.z > interPosTop.z) {
		return false;
	}
	return true;
}

__global__ void d_render(uint *d_output, uint imageW, uint imageH,
		float density, float brightness, float transferOffset,
		float transferScale, int queryMethod, int3 volumeSize,
		cudaSurfaceObject_t flexBlockSurfObj) {
	const int maxSteps = 500;
	const float tstep = 0.01f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH))
		return;

	float u = (x / (float) imageW) * 2.0f - 1.0f;
	float v = (y / (float) imageH) * 2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(
			mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit)
		return;

	if (tnear < 0.0f)
		tnear = 0.0f; // clamp to near plane

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d * tnear;
	float3 step = eyeRay.d * tstep;

	// interpolation: [-1, 1] -> [0, 1] -> [0, volumeSize] -> floor or ceil -> [0, 1]
	// interPos[8] index:
	// 2: (0, 1, 0)		3: (1, 1, 0)
	// 0: (0, 0, 0)		1: (1, 0, 0)
	// 6: (0, 1, 1)		7: (1, 1, 1)
	// 4: (0, 0, 1)		5: (1, 0, 1)
	float3 interPos[8];

	interPos[0].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[0].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[0].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[1].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[1].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[1].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[2].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[2].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[2].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[3].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[3].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[3].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[4].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[4].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[4].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[5].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[5].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[5].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[6].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[6].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[6].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	interPos[7].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x) / volumeSize.x;
	interPos[7].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y) / volumeSize.y;
	interPos[7].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z) / volumeSize.z;

	// calculate interpolated mean
	float MaxHistogram = 0.0217;
	float MinHistogram = 0.0;
	float binWidth = (MaxHistogram - MinHistogram) / (float) nBins;
	float mean[8];
	for (int j = 0; j < 8; j++) {
		mean[j] = 0;
		int index = tex3D(tex, interPos[j].x, interPos[j].y, interPos[j].z);
		for (int i = 0; i < nBins; i++) {
			float p = tex2DLayered(histogramTex, (float) i,
					(float) (index % 2500), (int) (index / 2500));
			mean[j] += p * (binWidth * i + binWidth / 2.0);
		}
	}

	/*
	 float3 tryShare[3][3][3];
	 for (int i = 0; i < 3; i++) {
	 tryShare[i][i][i] = make_float3((float)i*i*i, ((float)i*i*i)+1.0, ((float)i*i*i)+2.0);
	 }
	 printf("tryShare[0][0][0]: %f %f %f\n", tryShare[0][0][0].x, tryShare[0][0][0].y, tryShare[0][0][0].z);
	 printf("tryShare[1][1][1]: %f %f %f\n", tryShare[1][1][1].x, tryShare[0][0][0].y, tryShare[0][0][0].z);
	 printf("tryShare[2][2][2]: %f %f %f\n", tryShare[2][2][2].x, tryShare[0][0][0].y, tryShare[0][0][0].z);
	 */

	int count1 = 0;
	int count2 = 0;
	for (int i = 0; i < maxSteps; i++) {
		count1 = i;
		float sample = 0.5;
		// read from 3D texture
		// remap position to [0, 1] coordinates
		int index = tex3D(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
				pos.z * 0.5f + 0.5f);

		//if (x == 200 && y == 200) {
		//	printf("Pos\t\t %f, %f, %f\n", pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f);
		//	printf("InterPosBot\t %f, %f, %f\n", interPos[0].x, interPos[0].y, interPos[0].z);
		//	printf("InterPosTop\t %f, %f, %f\n", interPos[7].x, interPos[7].y, interPos[7].z);
		//}

		if (queryMethod == 7) {
			if (!inInterpolation(pos, interPos[0], interPos[7], x, y)) {
				count2++;
				interPos[0].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[0].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[0].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[1].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[1].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[1].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[2].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[2].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[2].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[3].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[3].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[3].z = floor((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[4].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[4].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[4].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[5].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[5].y = floor((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[5].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[6].x = floor((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[6].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[6].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				interPos[7].x = ceil((pos.x * 0.5f + 0.5f) * volumeSize.x)
						/ volumeSize.x;
				interPos[7].y = ceil((pos.y * 0.5f + 0.5f) * volumeSize.y)
						/ volumeSize.y;
				interPos[7].z = ceil((pos.z * 0.5f + 0.5f) * volumeSize.z)
						/ volumeSize.z;

				for (int j = 0; j < 8; j++) {
					mean[j] = 0;
					int index = tex3D(tex, interPos[j].x, interPos[j].y,
							interPos[j].z);
					for (int i = 0; i < nBins; i++) {
						float p = tex2DLayered(histogramTex, (float) i,
								(float) (index % 2500), (int) (index / 2500));
						mean[j] += p * (binWidth * i + binWidth / 2.0);
					}
				}
			}

			float xd = (pos.x * 0.5f + 0.5f - interPos[0].x)
					/ (interPos[1].x - interPos[0].x);
			float yd = (pos.y * 0.5f + 0.5f - interPos[0].y)
					/ (interPos[2].y - interPos[0].y);
			float zd = (pos.z * 0.5f + 0.5f - interPos[0].z)
					/ (interPos[4].z - interPos[0].z);
			float mean00 = mean[0] * (1.0 - xd) + mean[1] * xd;
			float mean10 = mean[2] * (1.0 - xd) + mean[3] * xd;
			float mean01 = mean[4] * (1.0 - xd) + mean[5] * xd;
			float mean11 = mean[6] * (1.0 - xd) + mean[7] * xd;
			float mean0 = mean00 * (1.0 - yd) + mean10 * yd;
			float mean1 = mean01 * (1.0 - yd) + mean11 * yd;
			float interMean = mean0 * (1.0 - zd) + mean1 * zd;
			sample = interMean * 50; // mean value is too small, so that we x50
		}

		/*
		 // fractal decoding
		 // Flip (if needed)->Shift->merge errors->normalize the frequencies
		 int4 codebookIndex = tex3D(codebookTex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
		 float originalTemplate[nBins];
		 float *currentTemplate;
		 int templateId = codebookIndex.x;
		 if (templateId < 0 || templateId > 622) { printf("Error! templateID: %d\n", templateId); }
		 int shift = codebookIndex.y;
		 int flipFlag = codebookIndex.z;
		 int NE = codebookIndex.w;
		 if (NE < 0 || NE > nBins) {
		 printf("Error NE!: %d\n", NE);
		 }

		 for (int i = 0; i < nBins; i++) {
		 originalTemplate[i] = tex2DLayered(templatesTex, (float) i, (float) templateId, 1);
		 if (originalTemplate[i] < 0 || originalTemplate[i] > 1) {
		 printf("Error! originalTemplate[%d]: %f\n", i, originalTemplate[i]);
		 }
		 }
		 currentTemplate = fractalDecoding(originalTemplate, flipFlag, shift);
		 for (int i = 0; i < nBins; i++) {
		 if (currentTemplate[i] < 0 || currentTemplate[i] > 1) {
		 printf("Error! currentTemplate[%d], %f\n", i, currentTemplate[i]);
		 }
		 }
		 // add NE
		 for (int i = 0; i < NE; i++) {
		 float2 error = tex2DLayered(errorsbookTex, (float) i, (float)(index % 2500), (int) (index / 2500) );
		 int errorIndex = (int)error.x;
		 if (errorIndex < 0 || errorIndex > nBins) {
		 printf("Error Index! %d\n", errorIndex);
		 }
		 float errorValue = error.y;
		 if (errorValue < -1 || errorValue > 1) {
		 printf("Error Value! %f\n", errorValue);
		 }
		 currentTemplate[errorIndex] += errorValue;	// TODO: in this step, we get wrong value
		 if (currentTemplate[errorIndex] < 0 ) { currentTemplate[errorIndex] = 0; }
		 if (currentTemplate[errorIndex] < 0 || currentTemplate[errorIndex] > 1) {
		 //printf("Error after NE: currentTemplate[%d]: %f\n", errorIndex, currentTemplate[errorIndex]);
		 }
		 }
		 // normalize currenTemplate
		 float tempTotal = 0;
		 for (int i = 0; i < nBins; i++) {
		 tempTotal += currentTemplate[i];
		 }
		 for (int i = 0; i < nBins; i++) {
		 currentTemplate[i] = currentTemplate[i] / tempTotal;
		 //if (currentTemplate[i] < 0 || currentTemplate[i] > 1) { printf("Error! currentTemplate[%d]: %f\n", i, currentTemplate[i]); }
		 }

		 float MaxHistogram1 = 0.0217;
		 float MinHistogram1 = 0.0;
		 float mean1 = 0;
		 float binWidth1 = (MaxHistogram1 - MinHistogram1) / (float)nBins;
		 for (int i = 0; i < nBins; i ++) {
		 //mean1 += currentTemplate[i] * ((float) i / (float) nBins) * MaxHistogram1;
		 //printf("original: %f\n", currentTemplate[i]);
		 mean1 += currentTemplate[i] * (binWidth1 * i + binWidth1 / 2.0);
		 }
		 //if (mean1 > 1 || mean1 < 0) { printf("Erro! mean1: %f\n", mean1); }
		 float variance1 = 0;
		 for (int i = 0; i < nBins; i++) {
		 //variance1 += currentTemplate[i] * (((float) i / (float) nBins) * MaxHistogram1 - mean1) * ( ((float) i / (float) nBins) * MaxHistogram1 - mean1);
		 variance1 += currentTemplate[i] * ((binWidth1 * i + binWidth1 / 2.0) - mean1) * ((binWidth1 * i + binWidth1 / 2.0) - mean1);
		 }
		 //float sample1 = variance1 * 1000000;	// variance value is too small, so that we x1000000
		 float sample1 = 0.5;
		 if (queryMethod == 4) {
		 //sample = mean1 * 50;
		 sample = mean1 / 0.0217;
		 }
		 if (queryMethod == 5) {
		 //sample = variance1 * 1000000;
		 sample = variance1 / 0.000021;
		 }
		 if (queryMethod == 6) {
		 // Compute negative entropy
		 float entropy = 0;
		 for( int i = 0; i < nBins; i++ )
		 {
		 entropy += ( currentTemplate[i] * ( currentTemplate[i] <= 0 ? 0 : ( log( currentTemplate[i] ) / log(2.0) ) ) );
		 //float probability = tex2DLayered(histogramTex, (float) i, (float)(index % 2500), (int) (index / 2500) );
		 //entropy += ( probability * ( probability <= 0 ? 0 : ( log( probability ) / log(2.0) ) ) );
		 }

		 // Change sign
		 entropy = -entropy;

		 // Normalize, if required
		 entropy /= ( log( (float)nBins ) / log( 2.0f ) );

		 sample = entropy;
		 }
		 */

		if (queryMethod == 3) {
			/*
			 // Compute negative entropy
			 float entropy = 0;
			 for( int i = 0; i<nBins; i++ )
			 {
			 float probability = tex2DLayered(histogramTex, (float) i, (float)(index % 2500), (int) (index / 2500) );
			 entropy += ( probability * ( probability <= 0 ? 0 : ( log( probability ) / log(2.0) ) ) );
			 }

			 // Change sign
			 entropy = -entropy;

			 // Normalize, if required
			 entropy /= ( log( (float)nBins ) / log( 2.0f ) );

			 sample = entropy;
			 */

			// optimization
			sample = tex3D(originalQueryTex, pos.x * 0.5f + 0.5f,
					pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f).z;
		}

		// calculate mean
		float MaxHistogram = 0.0217;
		float mean = 0;
		for (int i = 0; i < nBins; i++) {
			float p = tex2DLayered(histogramTex, (float) i,
					(float) (index % 2500), (int) (index / 2500));
			mean += p * ((float) i / (float) nBins) * MaxHistogram;
		}
		if (queryMethod == 1) {
			/*
			 sample = mean * 50;   // mean value is too small, so that we x50
			 */

			// optimization
			sample = tex3D(originalQueryTex, pos.x * 0.5f + 0.5f,
					pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f).x;
		}

		if (queryMethod == 2) {
			/*
			 // calculate variance
			 float variance = 0;
			 for (int i = 0; i < nBins; i++) {
			 float p = tex2DLayered(histogramTex, (float) i, (float)(index % 2500), (int) (index / 2500) );
			 variance += p * (((float) i / (float) nBins) * MaxHistogram - mean) * ( ((float) i / (float) nBins) * MaxHistogram - mean);
			 }
			 sample = variance * 1000000;	// variance value is too small, so that we x1000000
			 */

			// optimization
			sample = tex3D(originalQueryTex, pos.x * 0.5f + 0.5f,
					pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f).y;
		}

		if (queryMethod == 4) {
			sample = tex3D(fractalQueryTex, pos.x * 0.5f + 0.5f,
					pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f).x;
		}

		if (queryMethod == 5) {
			sample = tex3D(fractalQueryTex, pos.x * 0.5f + 0.5f,
					pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f).y;
		}

		if (queryMethod == 6) {
			sample = tex3D(fractalQueryTex, pos.x * 0.5f + 0.5f,
					pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f).z;
		}

		if (queryMethod == 8) {
			float4 x = tex3D(flexBlockTex, (pos.x * 0.5f + 0.5f) * nFlexBlockX,
					(pos.y * 0.5f + 0.5f) * nFlexBlockY,
					(pos.z * 0.5f + 0.5f) * nFlexBlockZ);
			//float4 x = tex3D(flexBlockTex, (pos.x*0.5f+0.5f), (pos.y*0.5f+0.5f), (pos.z*0.5f+0.5f));
			//float4 x;
			//surf3Dread(&x, flexBlockSurfObj, (pos.x*0.5f+0.5f)*3.0*sizeof(float4), (pos.y*0.5f+0.5f)*3.0, (pos.z*0.5f+0.5f)*3.0, cudaBoundaryModeClamp);
			sample = x.z;
		}

		if (queryMethod == 9) {
			float4 x = tex3D(flexBlockTex, (pos.x * 0.5f + 0.5f) * nFlexBlockX,
					(pos.y * 0.5f + 0.5f) * nFlexBlockY,
					(pos.z * 0.5f + 0.5f) * nFlexBlockZ);
			//float4 x;
			//surf3Dread(&x, flexBlockSurfObj, (pos.x*0.5f+0.5f)*3.0*sizeof(float4), (pos.y*0.5f+0.5f)*3.0, (pos.z*0.5f+0.5f)*3.0, cudaBoundaryModeClamp);
			sample = x.x;
		}

		if (queryMethod == 0) {
			float4 x = tex3D(flexBlockTex, (pos.x * 0.5f + 0.5f) * nFlexBlockX,
					(pos.y * 0.5f + 0.5f) * nFlexBlockY,
					(pos.z * 0.5f + 0.5f) * nFlexBlockZ);
			//float4 x;
			//surf3Dread(&x, flexBlockSurfObj, (pos.x*0.5f+0.5f)*3.0*sizeof(float4), (pos.y*0.5f+0.5f)*3.0, (pos.z*0.5f+0.5f)*3.0, cudaBoundaryModeClamp);
			sample = x.y;
		}

		// lookup in transfer function texture
		float4 col = tex1D(transferTex,
				(sample - transferOffset) * transferScale);
		col.w *= density;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col * (1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar)
			break;

		pos += step;
	}

	if (x == 200 && y == 200) {
		//	printf("Count1: %d   Count2: %d\n", count1, count2);
	}

	sum *= brightness;

	// write output color
	d_output[y * imageW + x] = rgbaFloatToInt(sum);
}

__device__ float4 originalHistogramData[nBlocks]; // x is mean, y is variance, z is entropy
__device__ float4 fractalHistogramData[nBlocks];

__global__ void d_basicDataProcessing() {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	int width = 50; // NOTE: here we hard code the size
	int height = 50;
	int depth = 10;

//	float x = (float) ix / (float) (width);
//	float y = (float) iy / (float) (height);
//	float z = (float) iz / (float) (depth);

	// calculate interpolated mean
	float MaxHistogram = 0.0217;
	float MinHistogram = 0.0;
	float binWidth = (MaxHistogram - MinHistogram) / (float) nBins;

	int index = ix + iy * width + iz * width * height;

	float mean = 0;
	for (int i = 0; i < nBins; i++) {
		float p = tex2DLayered(histogramTex, (float) i, (float) (index % 2500),
				(int) (index / 2500));
		mean += p * (binWidth * i + binWidth / 2.0);
	}

	float variance = 0;
	for (int i = 0; i < nBins; i++) {
		float p = tex2DLayered(histogramTex, (float) i, (float) (index % 2500),
				(int) (index / 2500));
		variance += p * (((float) i / (float) nBins) * MaxHistogram - mean)
				* (((float) i / (float) nBins) * MaxHistogram - mean);
	}

	// normalize mean and variance
	mean /= 0.0217;
	variance /= 0.000021;

	float entropy = 0;
	for (int i = 0; i < nBins; i++) {
		float probability = tex2DLayered(histogramTex, (float) i,
				(float) (index % 2500), (int) (index / 2500));
		entropy += (probability
				* (probability <= 0 ? 0 : (log(probability) / log(2.0))));
	}
	entropy = -entropy; // Change sign
	entropy /= (log((float) nBins) / log(2.0f)); // Normalize, if required

	originalHistogramData[ix + iy * width + iz * width * height].x = mean;
	originalHistogramData[ix + iy * width + iz * width * height].y = variance;
	originalHistogramData[ix + iy * width + iz * width * height].z = entropy;

	// fractal decoding
	// Flip (if needed)->Shift->merge errors->normalize the frequencies
	int4 codebookIndex = tex3D(codebookTex, ix, iy, iz);
	float originalTemplate[nBins];
	float *currentTemplate = 0;
	int templateId = codebookIndex.x;
	if (templateId < 0 || templateId > 622) {
		printf("Error! templateID: %d\n", templateId);
	}
	int shift = codebookIndex.y;
	int flipFlag = codebookIndex.z;
	int NE = codebookIndex.w;
	if (NE < 0 || NE > nBins) {
		printf("Error NE!: %d\n", NE);
	}

	for (int i = 0; i < nBins; i++) {
		originalTemplate[i] = tex2DLayered(templatesTex, (float) i,
				(float) templateId, 1);
		if (originalTemplate[i] < 0 || originalTemplate[i] > 1) {
			printf("Error! originalTemplate[%d]: %f\n", i, originalTemplate[i]);
		}
	}
	currentTemplate = fractalDecoding(originalTemplate, flipFlag, shift);
	for (int i = 0; i < nBins; i++) {
		if (currentTemplate[i] < 0 || currentTemplate[i] > 1) {
			printf("Error! currentTemplate[%d], %f\n", i, currentTemplate[i]);
		}
	}

	// add NE
	for (int i = 0; i < NE; i++) {
		float2 error = tex2DLayered(errorsbookTex, (float) i,
				(float) (index % 2500), (int) (index / 2500));
		int errorIndex = (int) error.x;
		if (errorIndex < 0 || errorIndex > nBins) {
			printf("Error Index! %d\n", errorIndex);
		}
		float errorValue = error.y;
		if (errorValue < -1 || errorValue > 1) {
			printf("Error Value! %f\n", errorValue);
		}
		currentTemplate[errorIndex] += errorValue; // TODO: in this step, we get wrong value
		if (currentTemplate[errorIndex] < 0) {
			currentTemplate[errorIndex] = 0;
		}
		if (currentTemplate[errorIndex] < 0
				|| currentTemplate[errorIndex] > 1) {
			//printf("Error after NE: currentTemplate[%d]: %f\n", errorIndex, currentTemplate[errorIndex]);
		}
	}

	// normalize currenTemplate
	float tempTotal = 0;
	for (int i = 0; i < nBins; i++) {
		tempTotal += currentTemplate[i];
	}
	for (int i = 0; i < nBins; i++) {
		if (tempTotal > 0) {
			currentTemplate[i] = currentTemplate[i] / tempTotal;
		}
		if (currentTemplate[i] < 0 || currentTemplate[i] > 1) {
			printf("Error! currentTemplate[%d]: %f\n", i, currentTemplate[i]);
		}
	}

	float MaxHistogram1 = 0.0217;
	float MinHistogram1 = 0.0;
	float mean1 = 0;
	float binWidth1 = (MaxHistogram1 - MinHistogram1) / (float) nBins;
	for (int i = 0; i < nBins; i++) {
		mean1 += currentTemplate[i] * (binWidth1 * i + binWidth1 / 2.0);
	}

	float variance1 = 0;
	for (int i = 0; i < nBins; i++) {
		variance1 += currentTemplate[i]
				* ((binWidth1 * i + binWidth1 / 2.0) - mean1)
				* ((binWidth1 * i + binWidth1 / 2.0) - mean1);
	}

	mean1 /= 0.0217;
	variance1 /= 0.000021;

	// Compute negative entropy
	float entropy1 = 0;
	for (int i = 0; i < nBins; i++) {
		entropy1 += (currentTemplate[i]
				* (currentTemplate[i] <= 0 ?
						0 : (log(currentTemplate[i]) / log(2.0))));
	}
	entropy1 = -entropy1; // Change sign
	entropy1 /= (log((float) nBins) / log(2.0f)); // Normalize, if required

	fractalHistogramData[ix + iy * width + iz * width * height].x = mean1;
	fractalHistogramData[ix + iy * width + iz * width * height].y = variance1;
	fractalHistogramData[ix + iy * width + iz * width * height].z = entropy1;
}

//__device__ int4* corner;
// NOTE: here we hard code the size
__device__ int4 corner[32768];


//__device__ Span** subSpan; // subSpans of each corner of each block
// NOTE: here we hard code the max subSpans, 16*16*16*8 corners, each corner 6*6*6 spans
__device__ Span subSpan[32768][216];

__device__ int* nSubSpan; // number of sub spans in that corner

//__device__ float** cornerSumHistogram; // the sum histogram of all spans of each corner
// NOTE: here we hard code the max block number 16*16*16, each block has 8 corners
__device__ float cornerSumHistogram[32768][flexNBin];	

// NOTE: here we hard code the size, because cudaMemcpyFromSymbol doesn't support dynamic allocated array
__device__ float4 flexBlockData[1000000]; // x is mean, y is variance, z is entropy

__global__ void d_divideBlock(int divPar, dim3 volumeDim) {
	// divPar is the size of each block specified by user
	if (divPar > volumeDim.x || divPar > volumeDim.y || divPar > volumeDim.z) {
		printf("wrong divPar! %d\n", divPar);
	}

	int nx = 0;
	int ny = 0;
	int nz = 0;

	int2* spanX;
	int2* spanY;
	int2* spanZ;

	if (volumeDim.x % divPar != 0) {
		nx = volumeDim.x / divPar + 1;
		spanX = new int2[nx];
	}
	else {
		nx = volumeDim.x / divPar;
		spanX = new int2[nx];
	}

	if (volumeDim.y % divPar != 0) {
		ny = volumeDim.y / divPar + 1;
		spanY = new int2[ny];
	}
	else {
		ny = volumeDim.y / divPar;
		spanY = new int2[ny];
	}

	if (volumeDim.z % divPar != 0) {
		nz = volumeDim.z / divPar + 1;
		spanZ = new int2[nz];
	}
	else {
		nz = volumeDim.z / divPar;
		spanZ = new int2[nz];
	}

	// flexBlock = new Span[nx * ny * nz];

	if (volumeDim.x % divPar != 0) {
		for (int i = 0; i < nx; i++) {
			spanX[i].x = 1 + i * divPar;
			spanY[i].x = 1 + i * divPar;
			spanZ[i].x = 1 + i * divPar;
			if (i != nx - 1) {
				spanX[i].y = (i + 1) * divPar;
				spanY[i].y = (i + 1) * divPar;
				spanZ[i].y = (i + 1) * divPar;
			} else {
				spanX[i].y = volumeDim.x;
				spanY[i].y = volumeDim.x;
				spanZ[i].y = volumeDim.x;
			}
		}
	} else {
		for (int i = 0; i < nx; i++) {
			spanX[i].x = 1 + i * divPar;
			spanY[i].x = 1 + i * divPar;
			spanZ[i].x = 1 + i * divPar;
			spanX[i].y = (i + 1) * divPar;
			spanY[i].y = (i + 1) * divPar;
			spanZ[i].y = (i + 1) * divPar;
		}
	}

	if (volumeDim.y % divPar != 0) {
		for (int i = 0; i < ny; i++) {
			spanX[i].x = 1 + i * divPar;
			spanY[i].x = 1 + i * divPar;
			spanZ[i].x = 1 + i * divPar;
			if (i != ny - 1) {
				spanX[i].y = (i + 1) * divPar;
				spanY[i].y = (i + 1) * divPar;
				spanZ[i].y = (i + 1) * divPar;
			} else {
				spanX[i].y = volumeDim.y;
				spanY[i].y = volumeDim.y;
				spanZ[i].y = volumeDim.y;
			}
		}
	} else {
		for (int i = 0; i < nx; i++) {
			spanX[i].x = 1 + i * divPar;
			spanY[i].x = 1 + i * divPar;
			spanZ[i].x = 1 + i * divPar;
			spanX[i].y = (i + 1) * divPar;
			spanY[i].y = (i + 1) * divPar;
			spanZ[i].y = (i + 1) * divPar;
		}
	}

	if (volumeDim.z % divPar != 0) {
		for (int i = 0; i < nz; i++) {
			spanX[i].x = 1 + i * divPar;
			spanY[i].x = 1 + i * divPar;
			spanZ[i].x = 1 + i * divPar;
			if (i != nz - 1) {
				spanX[i].y = (i + 1) * divPar;
				spanY[i].y = (i + 1) * divPar;
				spanZ[i].y = (i + 1) * divPar;
			} else {
				spanX[i].y = volumeDim.z;
				spanY[i].y = volumeDim.z;
				spanZ[i].y = volumeDim.z;
			}
		}
	} else {
		for (int i = 0; i < nz; i++) {
			spanX[i].x = 1 + i * divPar;
			spanY[i].x = 1 + i * divPar;
			spanZ[i].x = 1 + i * divPar;
			spanX[i].y = (i + 1) * divPar;
			spanY[i].y = (i + 1) * divPar;
			spanZ[i].y = (i + 1) * divPar;
		}
	}

	for (int z = 0; z < nx; z++) {
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nz; x++) {
				flexBlock[z * nx * ny + y * nx + x].low.x = spanX[x].x;
				flexBlock[z * nx * ny + y * nx + x].low.y = spanY[y].x;
				flexBlock[z * nx * ny + y * nx + x].low.z = spanZ[z].x;
				flexBlock[z * nx * ny + y * nx + x].high.x = spanX[x].y;
				flexBlock[z * nx * ny + y * nx + x].high.y = spanY[y].y;
				flexBlock[z * nx * ny + y * nx + x].high.z = spanZ[z].y;
			}
		}
	}
	nFlexBlock = nx * ny * nz;
	nFlexBlockX = nx;
	nFlexBlockY = ny;
	nFlexBlockZ = nz;
	printf("nFlexBlock: %d\n", nFlexBlock);
	printf("nx %d, ny %d, nz %d\n", nx, ny, nz);
}

__global__ void d_computeBlock() {
	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		printf("enter d_computeBlock()\n\n"); 
	}

	int n = blockIdx.x;
	float blockHistogram[flexNBin];

	for (int s = 0; s < flexNBin; s++) {

		blockHistogram[s] = cornerSumHistogram[n * 8 + 0][s] + cornerSumHistogram[n * 8 + 3][s]
				+ cornerSumHistogram[n * 8 + 4][s] + cornerSumHistogram[n * 8 + 7][s]
				- cornerSumHistogram[n * 8 + 1][s] - cornerSumHistogram[n * 8 + 2][s]
				- cornerSumHistogram[n * 8 + 5][s] - cornerSumHistogram[n * 8 + 6][s];
		if (blockHistogram[s] < 0) {
			blockHistogram[s] = 0;
		} // NOTE: here we just make sure it has no negative value

	}

	// normalize blockHistogram
//	Span currentSpan = flexBlock[n];
	// int weight = (currentSpan.high.x - currentSpan.low.x + 1)
	// 		* (currentSpan.high.y - currentSpan.low.y + 1)
	// 		* (currentSpan.high.z - currentSpan.low.z + 1);

	float totalBlockHistogram = 0;
	for (int s = 0; s < flexNBin; s++) {
		totalBlockHistogram += blockHistogram[s];
	}

	if (totalBlockHistogram <= 0) {
		printf("block %d: totalBlockHistogram = %f\n", n, totalBlockHistogram);
		//printf("block %d histogram:\n", n);
		//for (int i = 0; i < flexNBin; i++) {
		//	printf("%.6f ", blockHistogram[i]);
		//}
	} else {

		for (int s = 0; s < flexNBin; s++) {
			blockHistogram[s] = blockHistogram[s] / totalBlockHistogram;
			if (blockHistogram[s] < 0) {
				blockHistogram[s] = 0;
			}
			if (blockHistogram[s] > 1) {
				blockHistogram[s] = 1;
			}
		}
	}

	// compute mean
	float MaxHistogram = 255.0;
	float MinHistogram = 0.0;
	float mean = 0;
	float binWidth = (MaxHistogram - MinHistogram) / (float) flexNBin;
	for (int i = 0; i < flexNBin; i++) {
		mean += blockHistogram[i] * (binWidth * i + binWidth / 2.0);
	}
	// TODO: think about how to normalize mean
	//printf("n = %2d, mean = %f\n", n, mean);

	float variance = 0;
	for (int i = 0; i < flexNBin; i++) {
		variance += blockHistogram[i] * ((binWidth * i + binWidth / 2.0) - mean)
				* ((binWidth * i + binWidth / 2.0) - mean);
	}
	//printf("n = %2d, variance = %f\n", n, variance);

	// normalize mean and variance
	//mean = (mean - 1.992187) / (20.272202 - 1.992188);
	//variance /= 2132.295410;

	//compute entropy
	float entropy = 0;
	for (int i = 0; i < flexNBin; i++) {
		float probability = blockHistogram[i];
		entropy += (probability
				* (probability <= 0 ? 0 : (log(probability) / log(2.0))));
	}
	// Change sign
	entropy = -entropy;
	// Normalize, if required
	entropy /= (log((float) flexNBin) / log(2.0f));

	flexBlockData[n].x = mean;
	flexBlockData[n].y = variance;
	flexBlockData[n].z = entropy;

	// printf("i %d, mean %f, variance %f, entropy %f\n", n, flexBlockData[n].x, flexBlockData[n].y, flexBlockData[n].z);

	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		printf("finish d_computeBlock()\n\n"); 
	}
}

__global__ void d_allocateSpace(int nBlocks) {
	// corner = new int4[nBlocks * 8];
	nSubSpan = new int[nBlocks * 8];
//	subSpan = new Span*[nBlocks * 8];
//	cornerHistogram = new float**[nBlocks * 8];
//	cornerSumHistogram = new float*[nBlocks * 8];
//	for (int i = 0; i < nBlocks * 8; i++) {
//		cornerSumHistogram[i] = new float[flexNBin];
//	}
	//flexBlockData = new float4[nBlocks];
	printf("nBlocks: %d\n", nBlocks);
}

// d_queryBlockNew is to get subSpan of each corner
__global__ void d_queryBlockNew(dim3 volumeDim, int blockSize) {

	int blockNumber = blockIdx.x;

	int4 SpanLow = flexBlock[blockNumber].low;
	int4 SpanHigh = flexBlock[blockNumber].high;

	int cornerId = threadIdx.x;

	switch (cornerId) {
	// 2: (0, 1, 0)		3: (1, 1, 0)
	// 0: (0, 0, 0)		1: (1, 0, 0)
	// 6: (0, 1, 1)		7: (1, 1, 1)
	// 4: (0, 0, 1)		5: (1, 0, 1)
	case 0:
		corner[blockNumber * 8 + 0].x =
				(SpanLow.x < SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 0].y =
				(SpanLow.y < SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 0].z =
				(SpanLow.z < SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 0].w = 0;
		break;
	case 1:
		corner[blockNumber * 8 + 1].x =
				(SpanLow.x > SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 1].y =
				(SpanLow.y < SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 1].z =
				(SpanLow.z < SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 1].w = 0;
		break;
	case 2:
		corner[blockNumber * 8 + 2].x =
				(SpanLow.x < SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 2].y =
				(SpanLow.y > SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 2].z =
				(SpanLow.z < SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 2].w = 0;
		break;
	case 3:
		corner[blockNumber * 8 + 3].x =
				(SpanLow.x > SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 3].y =
				(SpanLow.y > SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 3].z =
				(SpanLow.z < SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 3].w = 0;
		break;
	case 4:
		corner[blockNumber * 8 + 4].x =
				(SpanLow.x < SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 4].y =
				(SpanLow.y < SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 4].z =
				(SpanLow.z > SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 4].w = 0;
		break;
	case 5:
		corner[blockNumber * 8 + 5].x =
				(SpanLow.x > SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 5].y =
				(SpanLow.y < SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 5].z =
				(SpanLow.z > SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 5].w = 0;
		break;
	case 6:
		corner[blockNumber * 8 + 6].x =
				(SpanLow.x < SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 6].y =
				(SpanLow.y > SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 6].z =
				(SpanLow.z > SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 6].w = 0;
		break;
	case 7:
		corner[blockNumber * 8 + 7].x =
				(SpanLow.x > SpanHigh.x) ? SpanLow.x : SpanHigh.x;
		corner[blockNumber * 8 + 7].y =
				(SpanLow.y > SpanHigh.y) ? SpanLow.y : SpanHigh.y;
		corner[blockNumber * 8 + 7].z =
				(SpanLow.z > SpanHigh.z) ? SpanLow.z : SpanHigh.z;
		corner[blockNumber * 8 + 7].w = 0;
		break;
	}

	// subSpanX[i].x is the lower span, subSpanX[i].y is the upper span
	// int2 *subSpanX = new int2[(int)ceil(log((float)volumeDim.x)/log(2.0))];
	// int2 *subSpanY = new int2[(int)ceil(log((float)volumeDim.y)/log(2.0))];
	// int2 *subSpanZ = new int2[(int)ceil(log((float)volumeDim.z)/log(2.0))];

	// NOTE: here we hard code the size
	int2 subSpanX[nLgTwo];
	int2 subSpanY[nLgTwo];
	int2 subSpanZ[nLgTwo];

	int nx = 0; // count how many spans generated in each dimension
	int ny = 0;
	int nz = 0;
	int n = 0; // total number of subSpans
	int x = corner[blockNumber * 8 + cornerId].x;
	int y = corner[blockNumber * 8 + cornerId].y;
	int z = corner[blockNumber * 8 + cornerId].z;

	for (int i = 0; i <= nLgTwo; i++) {
		if ((x & (~(1 << i))) != x) {
			subSpanX[nx].y = x;
			x &= ~(1 << i);
			subSpanX[nx].x = x + 1;
			//printf("subSpanX[%d]: low %d, high %d\n", nx, subSpanX[nx].x, subSpanX[nx].y);
			nx++;
		}
		if (x == 0) {
			break;
		}
	}
	for (int i = 0; i <= nLgTwo; i++) {
		if ((y & (~(1 << i))) != y) {
			subSpanY[ny].y = y;
			y &= ~(1 << i);
			subSpanY[ny].x = y + 1;
			//printf("subSpanY[%d]: low %d, high %d\n", ny, subSpanY[ny].x, subSpanY[ny].y);
			ny++;
		}
		if (y == 0) {
			break;
		}
	}
	for (int i = 0; i <= nLgTwo; i++) { // NOTE: here 6 = log64
		if ((z & (~(1 << i))) != z) {
			subSpanZ[nz].y = z;
			z &= ~(1 << i);
			subSpanZ[nz].x = z + 1;
			//printf("subSpanZ[%d]: low %d, high %d\n", nz, subSpanZ[nz].x, subSpanZ[nz].y);
			nz++;
		}
		if (z == 0) {
			break;
		}
	}

	n = nx * ny * nz;
	nSubSpan[blockNumber * 8 + cornerId] = n;

	//printf("block %d, nSubSpan %d\n", blockNumber * 8 + cornerId, n);

//	subSpan[blockNumber * 8 + cornerId] = new Span[n];
//	cornerHistogram[blockNumber * 8 + cornerId] = new float*[n];

//	for (int i = 0; i < n; i++) {
//		cornerHistogram[blockNumber * 8 + cornerId][i] = new float[flexNBin];
//	}
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				subSpan[blockNumber * 8 + cornerId][i * ny * nz + j * nz + k].low.x =
						subSpanX[i].x;
				subSpan[blockNumber * 8 + cornerId][i * ny * nz + j * nz + k].low.y =
						subSpanY[j].x;
				subSpan[blockNumber * 8 + cornerId][i * ny * nz + j * nz + k].low.z =
						subSpanZ[k].x;
				subSpan[blockNumber * 8 + cornerId][i * ny * nz + j * nz + k].high.x =
						subSpanX[i].y;
				subSpan[blockNumber * 8 + cornerId][i * ny * nz + j * nz + k].high.y =
						subSpanY[j].y;
				subSpan[blockNumber * 8 + cornerId][i * ny * nz + j * nz + k].high.z =
						subSpanZ[k].y;
			}
		}
	}

}

// d_querySpanNew is to get histogram of each span
__global__ void d_querySpanNew() {

	__shared__ float sharedCornerHistogram[flexNBin];
	for (int i = 0; i < flexNBin; i++) {
		atomicExch(&sharedCornerHistogram[i], 0);
	}

	__syncthreads();

	int cId = blockIdx.x; // corner Id, NOTE: cId not only represents corner, but also represents block
	int sId = threadIdx.x; // span Id
	if (sId >= nSubSpan[cId]) {
		return;
	}
	if (nSubSpan[cId] > 1000) {
		printf("This block has too many spans! cId = %d\n", cId);
	}

	Span currentSpan = subSpan[cId][sId];
	int4 currentFractal; // current fractal encoded histogram
	int indexError = 0; // index of NE in errorTexture

	int weight = (currentSpan.high.x - currentSpan.low.x + 1)
			* (currentSpan.high.y - currentSpan.low.y + 1)
			* (currentSpan.high.z - currentSpan.low.z + 1);

	if (weight <= 0) {
		printf("Wrong! weight = %d\n", weight);
	}

	// NOTE: Here we need to use a threshold to distinguish whether this span is fractal encoded or simple histogram
	if (d_spanSize(currentSpan) >= 8) {
		// look up fractal codebook
		int found = 0;
		for (int iz = 0; iz < nDimension / 2; iz++) {
			for (int iy = 0; iy < nDimension; iy++) {
				for (int ix = 0; ix < nDimension; ix++) { // NOTE: the fuel dataset size is 64x64x64, but half of them are encoded by fracal and half by simple histogram, so that each texture has 64x64x32 size
					int4 spanLow = tex3D(codebookSpanLowTex, ix, iy, iz);
					int4 spanHigh = tex3D(codebookSpanHighTex, ix, iy, iz);
					if ((spanLow.x == currentSpan.low.x)
							&& (spanLow.y == currentSpan.low.y)
							&& (spanLow.z == currentSpan.low.z)
							&& (spanHigh.x == currentSpan.high.x)
							&& (spanHigh.y == currentSpan.high.y)
							&& (spanHigh.z == currentSpan.high.z)) {
						currentFractal = tex3D(flexibleCodebookTex, ix, iy, iz);
						// TODO: add NE
						//indexError = ix * iy * iz;
						// NOTE: how to index error list?
						indexError = ix + iy * nDimension
								+ iz * nDimension * nDimension;
						found = 1;
						break;
					}
				}
			}
		} // end look up fractal codebook
		if (found == 0) {
			printf(
					"didn't find fractal: cId %d, sId %2d, spanLow(%2d, %2d, %2d), spanHigh(%2d, %2d, %2d)\n",
					cId, sId, currentSpan.low.x, currentSpan.low.y,
					currentSpan.low.z, currentSpan.high.x, currentSpan.high.y,
					currentSpan.high.z);
		}

		// fractal decoding
		// Flip (if needed)->Shift->merge errors->normalize the frequencies
		float originalTemplate[flexNBin];
		float *currentTemplate;
		int templateId = currentFractal.x;
		if (templateId < 0 || templateId > nTemplate) {
			printf("Error! templateID: %d\n", templateId);
		}
		int shift = currentFractal.y;
		int flipFlag = currentFractal.z;
		int NE = currentFractal.w;
		if (NE < 0 || NE > flexNBin) {
			printf("Error NE!: %d\n", NE);
		}
		for (int i = 0; i < flexNBin; i++) {
			originalTemplate[i] = tex2DLayered(flexibleTemplatesTex, (float) i,
					(float) templateId, 1);
			if (originalTemplate[i] < 0 || originalTemplate[i] > 1) {
				printf("Error! originalTemplate[%d]: %f\n", i,
						originalTemplate[i]);
			}
		}
		currentTemplate = flexibleFractalDecoding(originalTemplate, flipFlag,
				shift);
		for (int i = 0; i < flexNBin; i++) {
			if (currentTemplate[i] < 0 || currentTemplate[i] > 1) {
				printf("Error! currentTemplate[%d], %f\n", i,
						currentTemplate[i]);
			}
		}
		// add NE
		for (int i = 0; i < NE; i++) {
			// TODO: find out why we could not look up texture and use if printf command
			float2 error = tex2DLayered(flexibleErrorsbookTex, (float) i,
					(float) (indexError % 2048), (int) (indexError / 2048));
			int errorIndex = (int) error.x;
			if (errorIndex < 0 || errorIndex > flexNBin) {
				printf("Error Index! %d\n", errorIndex);
			}
			float errorValue = error.y;
			if (errorValue < -1 || errorValue > 1) {
				printf("Error Value! %f\n", errorValue);
			}
			// TODO: find out why next line makes program crush
			currentTemplate[errorIndex] += errorValue; // TODO: in this step, we get wrong value
			if (currentTemplate[errorIndex] < 0) {
				currentTemplate[errorIndex] = 0;
			}
		}
		// normalize currenTemplate
		float tempTotal = 0;
		for (int i = 0; i < flexNBin; i++) {
			tempTotal += currentTemplate[i];
		}
		for (int i = 0; i < flexNBin; i++) {
			currentTemplate[i] = currentTemplate[i] / tempTotal;
			if (currentTemplate[i] < 0 || currentTemplate[i] > 1) {
				printf("Error! currentTemplate[%d]: %f\n", i,
						currentTemplate[i]);
			}
		}

		// copy currentTemplate into cornerHistogram
		for (int countCopy = 0; countCopy < flexNBin; countCopy++) {
			atomicAdd(&sharedCornerHistogram[countCopy], currentTemplate[countCopy] * weight);
			// cornerHistogram[cId][sId][countCopy] = currentTemplate[countCopy];
		}

		// test currentTemplate
		float sumCT = 0;
		for (int i = 0; i < flexNBin; i++) {
			// sumCT += cornerHistogram[cId][sId][i];
			sumCT += currentTemplate[i];
		}
		if (sumCT > 1.000001 || sumCT < 0.999999) {
			printf("cId %d, sId %2d, sumCT: %f\n", cId, sId, sumCT);
		}

	} else {
		// look up simple histogram

		// NOTE: when we are doing bitwise operation, we assume range (1 ~ 64), but the simple span data range is (0 ~ 63)
		// NOTE: but the fractal span range is still (1 ~ 64)
		currentSpan.low.x -= 1;
		currentSpan.low.y -= 1;
		currentSpan.low.z -= 1;
		currentSpan.high.x -= 1;
		currentSpan.high.y -= 1;
		currentSpan.high.z -= 1;

		int currentSimpleCount = 0;
		int indexSimpleHistogram = 0;

		float currentSimpleHistogram[flexNBin];
		// initialize
		for (int i = 0; i < flexNBin; i++) {
			currentSimpleHistogram[i] = 0;
		}

		int found = 0;

		for (int iz = 0; iz < nDimension / 2; iz++) {
			for (int iy = 0; iy < nDimension; iy++) {
				for (int ix = 0; ix < nDimension; ix++) {
					int4 spanLow = tex3D(simpleSpanLowTex, ix, iy, iz);
					int4 spanHigh = tex3D(simpleSpanHighTex, ix, iy, iz);

					if (spanLow.x == currentSpan.low.x
							&& spanLow.y == currentSpan.low.y
							&& spanLow.z == currentSpan.low.z
							&& spanHigh.x == currentSpan.high.x
							&& spanHigh.y == currentSpan.high.y
							&& spanHigh.z == currentSpan.high.z) {
						currentSimpleCount = tex3D(simpleCountTex, ix, iy, iz);
						indexSimpleHistogram = ix + iy * nDimension
								+ iz * nDimension * nDimension;
						found = 1;
						break;
					}
				}
			}
		} // end looking up simpleHistogram
		if (found == 0) {
			printf(
					"didn't find simple! cId %d, sId %2d, simpleLow(%2d, %2d, %2d), simpleHigh(%2d, %2d, %2d)\n",
					cId, sId, currentSpan.low.x, currentSpan.low.y,
					currentSpan.low.z, currentSpan.high.x, currentSpan.high.y,
					currentSpan.high.z);
		}
		for (int ih = 0; ih < currentSimpleCount; ih++) {
			float2 current = tex2DLayered(simpleHistogramTex, ih,
					indexSimpleHistogram % 2048, indexSimpleHistogram / 2048);
			currentSimpleHistogram[(int) current.x] = current.y;
		}
		for (int countCopy = 0; countCopy < flexNBin; countCopy++) {
			atomicAdd(&sharedCornerHistogram[countCopy], currentSimpleHistogram[countCopy] * weight);
			// cornerHistogram[cId][sId][countCopy] =
			// 		currentSimpleHistogram[countCopy];
		}

		// test currentTemplate
		float sumCT = 0;
		for (int i = 0; i < flexNBin; i++) {
			// sumCT += cornerHistogram[cId][sId][i];
			sumCT += currentSimpleHistogram[i];
		}
		if (sumCT > 1.000001 || sumCT < 0.999999) {
			printf("cId %d, sId %2d, sumCT: %f\n", cId, sId, sumCT);
		}

	} // end look up simple codebook

	__syncthreads();

	for (int i = 0; i < flexNBin; i++) {
		atomicExch(&cornerSumHistogram[cId][i], sharedCornerHistogram[i]);
	}

	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		printf("finish d_querySpanNew()\n\n"); 
	}
}

__global__ void d_testFlexBlockData(int nBlocks) {
	for (int i = 0; i < nBlocks; i++) {
		printf("block i = %d, mean = %f, variance = %f, entropy = %f\n", i, flexBlockData[i].x, flexBlockData[i].y, flexBlockData[i].z);
	}
}

__device__ float4 tryArray[4];

__global__ void d_testDynamicCopy() {
	//int a = 3;
	//tryArray = new float4[a];
	tryArray[0] = make_float4(0.1, 0.2, 0.3, 0.0);
	tryArray[1] = make_float4(0.4, 0.5, 0.6, 0.0);
	tryArray[2] = make_float4(0.7, 0.8, 0.9, 0.0);
	printf("tryArray[1] = %f, %f, %f\n", tryArray[1].x, tryArray[1].y, tryArray[1].z);
}

void tryCopy() {
	int a = 3;
	float4* h_tryArray = new float4[a];
	checkCudaErrors(
			cudaMemcpyFromSymbol(h_tryArray, tryArray,
					a * sizeof(float4)));
	printf("h_tryArray[1] = %f, %f, %f\n", h_tryArray[1].x, h_tryArray[1].y, h_tryArray[1].z);
}

void bindToTex() {

	// copy data from device to host

	int h_nFlexBlock; // how many blocks in this volume
	int h_nFlexBlockX;
	int h_nFlexBlockY;
	int h_nFlexBlockZ;
	cudaMemcpyFromSymbol(&h_nFlexBlock, nFlexBlock, sizeof(int));
	cudaMemcpyFromSymbol(&h_nFlexBlockX, nFlexBlockX, sizeof(int));
	cudaMemcpyFromSymbol(&h_nFlexBlockY, nFlexBlockY, sizeof(int));
	cudaMemcpyFromSymbol(&h_nFlexBlockZ, nFlexBlockZ, sizeof(int));
	printf("x %d, y %d, z %d, total %d\n", h_nFlexBlockX, h_nFlexBlockY,
			h_nFlexBlockZ, h_nFlexBlock);

	float4 *h_flexBlockData = new float4[h_nFlexBlock];
	checkCudaErrors(
			cudaMemcpyFromSymbol(h_flexBlockData, flexBlockData,
					h_nFlexBlock * sizeof(float4)));

	// find max and min value;
	float meanMax = 0;
	float meanMin = 1000;
	float varianceMax = 0;
	float varianceMin = 1000;
	float entropyMax = 0;
	float entropyMin = 1000;
	for (int i = 0; i < h_nFlexBlock; i++) {
		if (meanMax < h_flexBlockData[i].x) {
			meanMax = h_flexBlockData[i].x;
		}
		if (meanMin > h_flexBlockData[i].x) {
			meanMin = h_flexBlockData[i].x;
		}
		if (varianceMax < h_flexBlockData[i].y) {
			varianceMax = h_flexBlockData[i].y;
		}
		if (varianceMin > h_flexBlockData[i].y) {
			varianceMin = h_flexBlockData[i].y;
		}
		if (entropyMax < h_flexBlockData[i].z) {
			entropyMax = h_flexBlockData[i].z;
		}
		if (entropyMin > h_flexBlockData[i].z) {
			entropyMin = h_flexBlockData[i].z;
		}
		//printf("block i = %2d, entropy = %f\n", i, h_flexBlockData[i].z);
	}
	printf("meanMax: %f, meanMin %f\n", meanMax, meanMin);
	printf("varianceMax: %f, varianceMin %f\n", varianceMax, varianceMin);
	printf("entropyMax: %f, entropyMin %f\n", entropyMax, entropyMin);

	/*
	 // write to a file
	 float *entropy = new float[h_nFlexBlock];
	 for (int i = 0; i < h_nFlexBlock; i++) {
	 entropy[i] = h_flexBlockData[i].x;
	 printf("entropy[%d] = %f\n", i, entropy[i]);
	 }
	 FILE * pFile;
	 pFile = fopen ("meanBlockSize8.bin", "wb");
	 fwrite (entropy , sizeof(float), h_nFlexBlock, pFile);
	 fclose (pFile);
	 */

	// NOTE: here we have to put the h_flexBlockData into a 32*32*32 array
	float4 *h_data = new float4[nMaxBlockDim * nMaxBlockDim * nMaxBlockDim];

	for (int z = 0; z < nMaxBlockDim; z++) {
		for (int y = 0; y < nMaxBlockDim; y++) {
			for (int x = 0; x < nMaxBlockDim; x++) {
				h_data[z * nMaxBlockDim * nMaxBlockDim + y * nMaxBlockDim + x] =
						make_float4(0.0, 0.0, 0.0, 0.0);
			}
		}
	}

	for (int x = 0; x < h_nFlexBlockZ; x++) {
		for (int y = 0; y < h_nFlexBlockY; y++) {
			for (int z = 0; z < h_nFlexBlockX; z++) {
				h_data[z * nMaxBlockDim * nMaxBlockDim + y * nMaxBlockDim + x] =
						h_flexBlockData[z * h_nFlexBlockX * h_nFlexBlockY
								+ y * h_nFlexBlockX + x];
			}
		}
	}

	// NOTE: I found that only const cudaExtent can be used when binding to texture reference

	// create 3D array
	cudaChannelFormatDesc channelDescFlexTex =
			cudaCreateChannelDesc<FlexTexType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_flexTexArray, &channelDescFlexTex,
					flexTexSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParamsFlexTex = { 0 };
	//copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	// we change the source from h_volume to volume
	copyParamsFlexTex.srcPtr = make_cudaPitchedPtr(h_data,
			flexTexSize.width * sizeof(FlexTexType), flexTexSize.width,
			flexTexSize.height);
	copyParamsFlexTex.dstArray = d_flexTexArray;
	copyParamsFlexTex.extent = flexTexSize;
	copyParamsFlexTex.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsFlexTex));

	// set texture parameters
	flexBlockTex.normalized = false; // NOTE: access with normalized texture coordinates
	//tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	//flexBlockTex.filterMode = cudaFilterModePoint;		//TODO: note we change it to nearest because we don't want histogram to be interpolated
	flexBlockTex.filterMode = cudaFilterModeLinear;
	flexBlockTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	flexBlockTex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(
			cudaBindTextureToArray(flexBlockTex, d_flexTexArray,
					channelDescFlexTex));



	delete[] h_flexBlockData;
}

__global__ void cleanPointers(int nBlocks) {
	// delete[] corner;

//	for (int i = 0; i < nBlocks * 8; ++i) {
//		delete[] subSpan[i];
//	}

//	for (int i = 0; i < nBlocks * 8; ++i) {
//		for (int j = 0; j < nSubSpan[i]; ++j) {
//			delete[] cornerHistogram[i][j];
//		}
//		delete[] cornerHistogram[i];
//	}

//	for (int i = 0; i < nBlocks * 8; ++i) {
//		delete[] cornerSumHistogram[i];
//	}

//	delete[] subSpan;
	delete[] nSubSpan;
//	delete[] cornerHistogram;
//	delete[] cornerSumHistogram;
	// delete[] flexBlock;
	//delete[] flexBlockData;
}

__global__ void d_testSharedMem() {
	__shared__ int s[4];
	int x = threadIdx.x;
	atomicAdd(&s[0], x);
	atomicAdd(&s[1], x * 2);
	atomicAdd(&s[2], x * 3);
	atomicAdd(&s[3], x * 4);
	__syncthreads();
	printf("s %d %d %d %d\n\n\n\n", s[0], s[1], s[2], s[3]);
}

extern "C" void dataProcessing() {

	int blockSize = 6;

	cudaEvent_t t1, t2, t3, t4, t5, t6, t7;
	float d1, d2, d3, d4, d5;

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);
	cudaEventCreate(&t3);
	cudaEventCreate(&t4);
	cudaEventCreate(&t5);
	cudaEventCreate(&t6);
	cudaEventCreate(&t7);

	cudaEventRecord(t1);
	d_divideBlock<<<1, 1>>>(blockSize, rawVolumeDim);
	cudaEventRecord(t2);

	int h_nFlexBlock; // how many blocks in this volume
	cudaMemcpyFromSymbol(&h_nFlexBlock, nFlexBlock, sizeof(int));
	printf("h_nFlexBlock: %d\n", h_nFlexBlock);

	cudaEventRecord(t3);
	d_allocateSpace<<<1, 1>>>(h_nFlexBlock);
	cudaEventRecord(t4);

	d_queryBlockNew<<<h_nFlexBlock, 8>>>(rawVolumeDim, blockSize);
	cudaEventRecord(t5);

	d_querySpanNew<<<h_nFlexBlock * 8, 1000>>>();
	cudaEventRecord(t6);

	// d_sumSpanHistogram<<<h_nFlexBlock, 8>>>();
	d_computeBlock<<<h_nFlexBlock, 1>>>();
	cudaEventRecord(t7);
	cudaEventSynchronize(t7);

	cudaEventElapsedTime(&d1, t1, t2);
	cudaEventElapsedTime(&d2, t3, t4);
	cudaEventElapsedTime(&d3, t4, t5);
	cudaEventElapsedTime(&d4, t5, t6);
	cudaEventElapsedTime(&d5, t6, t7);

	printf("d_divideBlock(): %f ms\n", d1);
	printf("d_allocateSpace(): %f ms\n", d2);
	printf("d_queryBlockNew(): %f ms\n", d3);
	printf("d_querySpanNew(): %f ms\n", d4);
	printf("d_computeBlock(): %f ms\n", d5);

	cudaDeviceSynchronize();

//	d_testDynamicCopy<<<1, 1>>>();
//	tryCopy();

//	d_testFlexBlockData<<<1, 1>>>(h_nFlexBlock);

	bindToTex();

	cleanPointers<<<1, 1>>>(h_nFlexBlock);

}

extern "C" void basicDataProcessing() {
	dim3 bSize(5, 5, 5);
	dim3 tSize(10, 10, 2);
	d_basicDataProcessing<<<bSize, tSize>>>();

	float4 h_basicQueryData[nBlocks];
	float4 h_fractalQueryData[nBlocks];
	cudaMemcpyFromSymbol(h_basicQueryData, originalHistogramData,
			sizeof(float4) * nBlocks);
	cudaMemcpyFromSymbol(h_fractalQueryData, fractalHistogramData,
			sizeof(float4) * nBlocks);

	/*
	 printf("test basicData\n");
	 for (int i = 0; i < nBlocks; i++) {
	 printf("i = %5d, mean = %1.6f, variance = %1.6f, entropy = %1.6f\n", i, h_basicQueryData[i].x, h_basicQueryData[i].y, h_basicQueryData[i].z);
	 }
	 */

	float varianceMax = 0;
	for (int i = 0; i < nBlocks; i++) {
		float variance = h_basicQueryData[i].y;
		if (varianceMax < variance) {
			varianceMax = variance;
		}
	}
	//printf("varianceMax = %f\n", varianceMax);

	// create 3D array
	cudaChannelFormatDesc channelDescOriginalQuery = cudaCreateChannelDesc<
			QueryType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_originalQueryArray, &channelDescOriginalQuery,
					originalVolumeSize));

	// create 3D array
	cudaChannelFormatDesc channelDescFractalQuery = cudaCreateChannelDesc<
			QueryType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_fractalQueryArray, &channelDescFractalQuery,
					originalVolumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParamsOriginalQuery = { 0 };
	//copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	// we change the source from h_volume to volume
	copyParamsOriginalQuery.srcPtr = make_cudaPitchedPtr(h_basicQueryData,
			originalVolumeSize.width * sizeof(QueryType),
			originalVolumeSize.width, originalVolumeSize.height);
	copyParamsOriginalQuery.dstArray = d_originalQueryArray;
	copyParamsOriginalQuery.extent = originalVolumeSize;
	copyParamsOriginalQuery.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsOriginalQuery));

	// copy data to 3D array
	cudaMemcpy3DParms copyParamsFractalQuery = { 0 };
	//copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	// we change the source from h_volume to volume
	copyParamsFractalQuery.srcPtr = make_cudaPitchedPtr(h_fractalQueryData,
			originalVolumeSize.width * sizeof(QueryType),
			originalVolumeSize.width, originalVolumeSize.height);
	copyParamsFractalQuery.dstArray = d_fractalQueryArray;
	copyParamsFractalQuery.extent = originalVolumeSize;
	copyParamsFractalQuery.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsFractalQuery));

	// set texture parameters
	originalQueryTex.normalized = true; // access with normalized texture coordinates
	originalQueryTex.filterMode = cudaFilterModeLinear; // linear interpolation
	//originalQueryTex.filterMode = cudaFilterModePoint;		//TODO: note we change it to nearest because we don't want histogram to be interpolated
	originalQueryTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	originalQueryTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters
	fractalQueryTex.normalized = true; // access with normalized texture coordinates
	fractalQueryTex.filterMode = cudaFilterModeLinear; // linear interpolation
	//fractalQueryTex.filterMode = cudaFilterModePoint;		//TODO: note we change it to nearest because we don't want histogram to be interpolated
	fractalQueryTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	fractalQueryTex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(
			cudaBindTextureToArray(originalQueryTex, d_originalQueryArray,
					channelDescOriginalQuery));

	// bind array to 3D texture
	checkCudaErrors(
			cudaBindTextureToArray(fractalQueryTex, d_fractalQueryArray,
					channelDescFractalQuery));
}

extern "C" void setTextureFilterMode(bool bLinearFilter) {
	tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C" void initCuda(float *h_histogram, cudaExtent volumeSize,
		cudaExtent histogramSize, int4 *h_codebook, cudaExtent codebookSize,
		float *h_templates, cudaExtent templatesSize, float2 *h_errorsbook,
		cudaExtent errorsbookSize, int4 *h_codebookSpanLow,
		int4 *h_codebookSpanHigh, int4 *h_flexibleCodebook,
		float2 *h_flexibleErrorsbook, int4 *h_simpleLow, int4 *h_simpleHigh,
		int *h_simpleCount, float2 *h_simpleHistogram,
		float *h_flexibleTemplates) {

	// create a volume data in which each voxel has a index number which can be used to lookup the histogram location
	int volume[nBlocks];
	for (int i = 0; i < nBlocks; i++) {
		volume[i] = i;
	}

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// create 3D array for histogram
	cudaChannelFormatDesc channelDescHistogram = cudaCreateChannelDesc<
			HistogramType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_histogramArray, &channelDescHistogram,
					histogramSize, cudaArrayLayered));

	// create 3D array for codebook
	cudaChannelFormatDesc channelDescCodebook = cudaCreateChannelDesc<
			CodebookType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_codebookArray, &channelDescCodebook,
					codebookSize));

	// create 3D array for errorsbook
	cudaChannelFormatDesc channelDescErrorsbook = cudaCreateChannelDesc<
			ErrorsbookType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_errorsbookArray, &channelDescErrorsbook,
					errorsbookSize, cudaArrayLayered));

	// create 3D array for templates
	cudaChannelFormatDesc channelDescTemplates = cudaCreateChannelDesc<
			TemplatesType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_templatesArray, &channelDescTemplates,
					templatesSize, cudaArrayLayered));

	////////////////////////////////////////////
	// flexible block size
	////////////////////////////////////////////

	// create 3D array for codebookSpanLow
	cudaChannelFormatDesc channelDescCodebookSpanLow = cudaCreateChannelDesc<
			SpanType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_codebookSpanLowArray,
					&channelDescCodebookSpanLow, flexibleVolumeSize));

	// create 3D array for codebookSpanHigh
	cudaChannelFormatDesc channelDescCodebookSpanHigh = cudaCreateChannelDesc<
			SpanType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_codebookSpanHighArray,
					&channelDescCodebookSpanHigh, flexibleVolumeSize));

	// create 3D array for flexibleCodebook
	cudaChannelFormatDesc channelDescFlexibleCodebook = cudaCreateChannelDesc<
			FlexibleCodebookType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_flexibleCodebookArray,
					&channelDescFlexibleCodebook, flexibleVolumeSize));

	// create 3D array for flexibleErrorsbook
	cudaChannelFormatDesc channelDescFlexibleErrorsbook = cudaCreateChannelDesc<
			FlexibleErrorsbookType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_flexibleErrorsbookArray,
					&channelDescFlexibleErrorsbook, flexibleHistogramSize,
					cudaArrayLayered));

	// create 3D array for simpleSpanLow
	cudaChannelFormatDesc channelDescSimpleSpanLow = cudaCreateChannelDesc<
			SpanType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_simpleSpanLowArray, &channelDescSimpleSpanLow,
					flexibleVolumeSize));

	// create 3D array for simpleSpanHigh
	cudaChannelFormatDesc channelDescSimpleSpanHigh = cudaCreateChannelDesc<
			SpanType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_simpleSpanHighArray,
					&channelDescSimpleSpanHigh, flexibleVolumeSize));

	// create 3D array for simpleCount
	cudaChannelFormatDesc channelDescSimpleCount = cudaCreateChannelDesc<
			SimpleCountType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_simpleCountArray, &channelDescSimpleCount,
					flexibleVolumeSize));

	// create 3D array for simpleHistogram
	cudaChannelFormatDesc channelDescSimpleHistogram = cudaCreateChannelDesc<
			SimpleHistogramType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_simpleHistogramArray,
					&channelDescSimpleHistogram, flexibleHistogramSize,
					cudaArrayLayered));

	// create 3D array for flexibleTemplates
	cudaChannelFormatDesc channelDescFlexibleTemplates = cudaCreateChannelDesc<
			TemplatesType>();
	checkCudaErrors(
			cudaMalloc3DArray(&d_flexibleTemplatesArray,
					&channelDescFlexibleTemplates, flexibleTemplatesSize,
					cudaArrayLayered));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	//copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	// we change the source from h_volume to volume
	copyParams.srcPtr = make_cudaPitchedPtr(volume,
			volumeSize.width * sizeof(VolumeType), volumeSize.width,
			volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// copy data to 3D array
	cudaMemcpy3DParms copyParamsHistogram = { 0 };
	copyParamsHistogram.srcPtr = make_cudaPitchedPtr(h_histogram,
			histogramSize.width * sizeof(HistogramType), histogramSize.width,
			histogramSize.height);
	copyParamsHistogram.dstArray = d_histogramArray;
	copyParamsHistogram.extent = histogramSize;
	copyParamsHistogram.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsHistogram));

	// copy data to 3D array of errorsbook
	cudaMemcpy3DParms copyParamsErrorsbook = { 0 };
	copyParamsErrorsbook.srcPtr = make_cudaPitchedPtr(h_errorsbook,
			errorsbookSize.width * sizeof(ErrorsbookType), errorsbookSize.width,
			errorsbookSize.height);
	copyParamsErrorsbook.dstArray = d_errorsbookArray;
	copyParamsErrorsbook.extent = errorsbookSize;
	copyParamsErrorsbook.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsErrorsbook));

	// copy data to 3D array of codebook
	cudaMemcpy3DParms copyParamsCodebook = { 0 };
	copyParamsCodebook.srcPtr = make_cudaPitchedPtr(h_codebook,
			codebookSize.width * sizeof(CodebookType), codebookSize.width,
			codebookSize.height);
	copyParamsCodebook.dstArray = d_codebookArray;
	copyParamsCodebook.extent = codebookSize;
	copyParamsCodebook.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsCodebook));

	// copy data to 3D array of templates
	cudaMemcpy3DParms copyParamsTemplates = { 0 };
	copyParamsTemplates.srcPtr = make_cudaPitchedPtr(h_templates,
			templatesSize.width * sizeof(TemplatesType), templatesSize.width,
			templatesSize.height);
	copyParamsTemplates.dstArray = d_templatesArray;
	copyParamsTemplates.extent = templatesSize;
	copyParamsTemplates.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsTemplates));

	////////////////////////////////////////////////
	// flexible block size
	///////////////////////////////////////////////

	// copy data to 3D array of codebookSpanLow
	cudaMemcpy3DParms copyParamsCodebookSpanLow = { 0 };
	copyParamsCodebookSpanLow.srcPtr = make_cudaPitchedPtr(h_codebookSpanLow,
			flexibleVolumeSize.width * sizeof(SpanType),
			flexibleVolumeSize.width, flexibleVolumeSize.height);
	copyParamsCodebookSpanLow.dstArray = d_codebookSpanLowArray;
	copyParamsCodebookSpanLow.extent = flexibleVolumeSize;
	copyParamsCodebookSpanLow.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsCodebookSpanLow));

	// copy data to 3D array of codebookSpanHigh
	cudaMemcpy3DParms copyParamsCodebookSpanHigh = { 0 };
	copyParamsCodebookSpanHigh.srcPtr = make_cudaPitchedPtr(h_codebookSpanHigh,
			flexibleVolumeSize.width * sizeof(SpanType),
			flexibleVolumeSize.width, flexibleVolumeSize.height);
	copyParamsCodebookSpanHigh.dstArray = d_codebookSpanHighArray;
	copyParamsCodebookSpanHigh.extent = flexibleVolumeSize;
	copyParamsCodebookSpanHigh.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsCodebookSpanHigh));

	// copy data to 3D array of flexibleCodebook
	cudaMemcpy3DParms copyParamsFlexibleCodebook = { 0 };
	copyParamsFlexibleCodebook.srcPtr = make_cudaPitchedPtr(h_flexibleCodebook,
			flexibleVolumeSize.width * sizeof(FlexibleCodebookType),
			flexibleVolumeSize.width, flexibleVolumeSize.height);
	copyParamsFlexibleCodebook.dstArray = d_flexibleCodebookArray;
	copyParamsFlexibleCodebook.extent = flexibleVolumeSize;
	copyParamsFlexibleCodebook.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsFlexibleCodebook));

	// copy data to 3D array of flexibleErrorsbook
	cudaMemcpy3DParms copyParamsFlexibleErrorsbook = { 0 };
	copyParamsFlexibleErrorsbook.srcPtr = make_cudaPitchedPtr(
			h_flexibleErrorsbook,
			flexibleHistogramSize.width * sizeof(FlexibleErrorsbookType),
			flexibleHistogramSize.width, flexibleHistogramSize.height);
	copyParamsFlexibleErrorsbook.dstArray = d_flexibleErrorsbookArray;
	copyParamsFlexibleErrorsbook.extent = flexibleHistogramSize;
	copyParamsFlexibleErrorsbook.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&copyParamsFlexibleErrorsbook));

	// copy data to 3D array of simpleLow
	cudaMemcpy3DParms copyParamsSimpleLow = { 0 };
	copyParamsSimpleLow.srcPtr = make_cudaPitchedPtr(h_simpleLow,
			flexibleVolumeSize.width * sizeof(SpanType),
			flexibleVolumeSize.width, flexibleVolumeSize.height);
	copyParamsSimpleLow.dstArray = d_simpleSpanLowArray;
	copyParamsSimpleLow.extent = flexibleVolumeSize;
	copyParamsSimpleLow.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsSimpleLow));

	// copy data to 3D array of simpleHigh
	cudaMemcpy3DParms copyParamsSimpleHigh = { 0 };
	copyParamsSimpleHigh.srcPtr = make_cudaPitchedPtr(h_simpleHigh,
			flexibleVolumeSize.width * sizeof(SpanType),
			flexibleVolumeSize.width, flexibleVolumeSize.height);
	copyParamsSimpleHigh.dstArray = d_simpleSpanHighArray;
	copyParamsSimpleHigh.extent = flexibleVolumeSize;
	copyParamsSimpleHigh.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsSimpleHigh));

	// copy data to 3D array of simpleCount
	cudaMemcpy3DParms copyParamsSimpleCount = { 0 };
	copyParamsSimpleCount.srcPtr = make_cudaPitchedPtr(h_simpleCount,
			flexibleVolumeSize.width * sizeof(SimpleCountType),
			flexibleVolumeSize.width, flexibleVolumeSize.height);
	copyParamsSimpleCount.dstArray = d_simpleCountArray;
	copyParamsSimpleCount.extent = flexibleVolumeSize;
	copyParamsSimpleCount.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsSimpleCount));

	// copy data to 3D array of simpleHistogram
	cudaMemcpy3DParms copyParamsSimpleHistogram = { 0 };
	copyParamsSimpleHistogram.srcPtr = make_cudaPitchedPtr(h_simpleHistogram,
			flexibleHistogramSize.width * sizeof(SimpleHistogramType),
			flexibleHistogramSize.width, flexibleHistogramSize.height);
	copyParamsSimpleHistogram.dstArray = d_simpleHistogramArray;
	copyParamsSimpleHistogram.extent = flexibleHistogramSize;
	copyParamsSimpleHistogram.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsSimpleHistogram));

	// copy data to 3D array of flexibleTemplates
	cudaMemcpy3DParms copyParamsFlexibleTemplates = { 0 };
	copyParamsFlexibleTemplates.srcPtr = make_cudaPitchedPtr(
			h_flexibleTemplates,
			flexibleTemplatesSize.width * sizeof(TemplatesType),
			flexibleTemplatesSize.width, flexibleTemplatesSize.height);
	copyParamsFlexibleTemplates.dstArray = d_flexibleTemplatesArray;
	copyParamsFlexibleTemplates.extent = flexibleTemplatesSize;
	copyParamsFlexibleTemplates.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParamsFlexibleTemplates));

	// set texture parameters
	tex.normalized = true; // access with normalized texture coordinates
	//tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.filterMode = cudaFilterModePoint; //TODO: note we change it to nearest because we don't want histogram to be interpolated
	tex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters
	histogramTex.normalized = false; // access with normalized texture coordinates
	histogramTex.filterMode = cudaFilterModePoint; //TODO: note we change it to nearest because we don't want histogram to be interpolated
	histogramTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	histogramTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for codebook
	//codebookTex.normalized = true;                      // NOTE: BE EXTREMELLY CAREFUL, if the ray marching through this 3D texture, we have to normalize this texture, because in the sample code the marching ray is re-mapped to [0, 1] coordinates
	codebookTex.normalized = false;
	codebookTex.filterMode = cudaFilterModePoint; // TODO: note we change it to nearest because we don't want histogram to be interpolated
	codebookTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
	codebookTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for templates
	templatesTex.normalized = false;
	templatesTex.filterMode = cudaFilterModePoint;
	templatesTex.addressMode[0] = cudaAddressModeClamp;
	templatesTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for errorsbook
	errorsbookTex.normalized = false;
	errorsbookTex.filterMode = cudaFilterModePoint;
	errorsbookTex.addressMode[0] = cudaAddressModeClamp;
	errorsbookTex.addressMode[1] = cudaAddressModeClamp;

	///////////////////////////////////////////////
	// flexible block size
	///////////////////////////////////////////////

	// set texture parameters for codebookSpanLowTex
	codebookSpanLowTex.normalized = false;
	codebookSpanLowTex.filterMode = cudaFilterModePoint;
	codebookSpanLowTex.addressMode[0] = cudaAddressModeClamp;
	codebookSpanLowTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for codebookSpanHighTex
	codebookSpanHighTex.normalized = false;
	codebookSpanHighTex.filterMode = cudaFilterModePoint;
	codebookSpanHighTex.addressMode[0] = cudaAddressModeClamp;
	codebookSpanHighTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for flexibleCodebookTex
	flexibleCodebookTex.normalized = false;
	flexibleCodebookTex.filterMode = cudaFilterModePoint;
	flexibleCodebookTex.addressMode[0] = cudaAddressModeClamp;
	flexibleCodebookTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for flexibleErrorsbookTex
	flexibleErrorsbookTex.normalized = false;
	flexibleErrorsbookTex.filterMode = cudaFilterModePoint;
	flexibleErrorsbookTex.addressMode[0] = cudaAddressModeClamp;
	flexibleErrorsbookTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for simpleSpanLowTex
	simpleSpanLowTex.normalized = false;
	simpleSpanLowTex.filterMode = cudaFilterModePoint;
	simpleSpanLowTex.addressMode[0] = cudaAddressModeClamp;
	simpleSpanLowTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for simpleSpanHighTex
	simpleSpanHighTex.normalized = false;
	simpleSpanHighTex.filterMode = cudaFilterModePoint;
	simpleSpanHighTex.addressMode[0] = cudaAddressModeClamp;
	simpleSpanHighTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for simpleCountTex
	simpleCountTex.normalized = false;
	simpleCountTex.filterMode = cudaFilterModePoint;
	simpleCountTex.addressMode[0] = cudaAddressModeClamp;
	simpleCountTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for simpleHistogramTex
	simpleHistogramTex.normalized = false;
	simpleHistogramTex.filterMode = cudaFilterModePoint;
	simpleHistogramTex.addressMode[0] = cudaAddressModeClamp;
	simpleHistogramTex.addressMode[1] = cudaAddressModeClamp;

	// set texture parameters for flexibleTemplatesTex
	flexibleTemplatesTex.normalized = false;
	flexibleTemplatesTex.filterMode = cudaFilterModePoint;
	flexibleTemplatesTex.addressMode[0] = cudaAddressModeClamp;
	flexibleTemplatesTex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

	// bind array to 2D layered texture
	checkCudaErrors(
			cudaBindTextureToArray(histogramTex, d_histogramArray,
					channelDescHistogram));

	// bind array to 3D texture of codebook
	checkCudaErrors(
			cudaBindTextureToArray(codebookTex, d_codebookArray,
					channelDescCodebook));

	// bind array to 2D layered texture of templates
	checkCudaErrors(
			cudaBindTextureToArray(templatesTex, d_templatesArray,
					channelDescTemplates));

	// bind array to 2D layered texture of errorsbook
	checkCudaErrors(
			cudaBindTextureToArray(errorsbookTex, d_errorsbookArray,
					channelDescErrorsbook));

	/////////////////////////////////////////////////////////////
	// flexible block size
	/////////////////////////////////////////////////////////////

	// bind array to 3D texture of codebookSpanLow
	checkCudaErrors(
			cudaBindTextureToArray(codebookSpanLowTex, d_codebookSpanLowArray,
					channelDescCodebookSpanLow));

	// bind array to 3D texture of codebookSpanHigh
	checkCudaErrors(
			cudaBindTextureToArray(codebookSpanHighTex, d_codebookSpanHighArray,
					channelDescCodebookSpanHigh));

	// bind array to 3D texture of flexibleCodebook
	checkCudaErrors(
			cudaBindTextureToArray(flexibleCodebookTex, d_flexibleCodebookArray,
					channelDescFlexibleCodebook));

	// bind array to 2D layered texture of flexibleErrorsbook
	checkCudaErrors(
			cudaBindTextureToArray(flexibleErrorsbookTex,
					d_flexibleErrorsbookArray, channelDescFlexibleErrorsbook));

	// bind array to 3D texture of simpleSpanLow
	checkCudaErrors(
			cudaBindTextureToArray(simpleSpanLowTex, d_simpleSpanLowArray,
					channelDescSimpleSpanLow));

	// bind array to 3D texture of simpleSpanHigh
	checkCudaErrors(
			cudaBindTextureToArray(simpleSpanHighTex, d_simpleSpanHighArray,
					channelDescSimpleSpanHigh));

	// bind array to 3D texture of simpleCount
	checkCudaErrors(
			cudaBindTextureToArray(simpleCountTex, d_simpleCountArray,
					channelDescSimpleCount));

	// bind array to 2D layered texture of simpleHistogram
	checkCudaErrors(
			cudaBindTextureToArray(simpleHistogramTex, d_simpleHistogramArray,
					channelDescSimpleHistogram));

	// bind array to 2D layered texture of flexibleTemplates
	checkCudaErrors(
			cudaBindTextureToArray(flexibleTemplatesTex,
					d_flexibleTemplatesArray, channelDescFlexibleTemplates));

	// create transfer function texture
	float4 transferFunc[] = { { 0.0, 0.0, 0.0, 0.0, }, { 1.0, 0.0, 0.0, 1.0, },
			{ 1.0, 0.5, 0.0, 1.0, }, { 1.0, 1.0, 0.0, 1.0, }, { 0.0, 1.0, 0.0,
					1.0, }, { 0.0, 1.0, 1.0, 1.0, }, { 0.0, 0.0, 1.0, 1.0, }, {
					1.0, 0.0, 1.0, 1.0, }, { 0.0, 0.0, 0.0, 0.0, }, };

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArray;
	checkCudaErrors(
			cudaMallocArray(&d_transferFuncArray, &channelDesc2,
					sizeof(transferFunc) / sizeof(float4), 1));
	checkCudaErrors(
			cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc,
					sizeof(transferFunc), cudaMemcpyHostToDevice));

	transferTex.filterMode = cudaFilterModeLinear;
	transferTex.normalized = true; // access with normalized texture coordinates
	transferTex.addressMode[0] = cudaAddressModeClamp; // wrap texture coordinates

	// Bind the array to the texture
	checkCudaErrors(
			cudaBindTextureToArray(transferTex, d_transferFuncArray,
					channelDesc2));

	for (int i = 0; i < 622; i++) {
		for (int j = 0; j < nBins; j++) {
			if (h_templates[i * nBins + j] < 0
					|| h_templates[i * nBins + j] > 1) {
				printf("ERROR! template(%d, %d): %f\n", i, j,
						h_templates[i * nBins + j]);
			}
			if (i == 32) {
				//printf("template(32, %d): %f\n", j, h_templates[i * nBins + j]);
			}
		}
	}
}

extern "C" void freeCudaBuffers() {
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_transferFuncArray));
	checkCudaErrors(cudaFreeArray(d_histogramArray));
	checkCudaErrors(cudaFreeArray(d_codebookArray));
	checkCudaErrors(cudaFreeArray(d_templatesArray));
	checkCudaErrors(cudaFreeArray(d_codebookTemplateIdArray));
	checkCudaErrors(cudaFreeArray(d_errorsbookArray));

	// flexible block size
	checkCudaErrors(cudaFreeArray(d_codebookSpanLowArray));
	checkCudaErrors(cudaFreeArray(d_codebookSpanHighArray));
	checkCudaErrors(cudaFreeArray(d_flexibleCodebookArray));
	checkCudaErrors(cudaFreeArray(d_flexibleErrorsbookArray));
	checkCudaErrors(cudaFreeArray(d_simpleSpanLowArray));
	checkCudaErrors(cudaFreeArray(d_simpleSpanHighArray));
	checkCudaErrors(cudaFreeArray(d_simpleCountArray));
	checkCudaErrors(cudaFreeArray(d_simpleHistogramArray));
	checkCudaErrors(cudaFreeArray(d_flexibleTemplatesArray));

	checkCudaErrors(cudaFreeArray(d_flexTexArray));

	checkCudaErrors(cudaFreeArray(d_originalQueryArray));
	checkCudaErrors(cudaFreeArray(d_fractalQueryArray));

}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output,
		uint imageW, uint imageH, float density, float brightness,
		float transferOffset, float transferScale, int queryMethod,
		cudaExtent volumeSize) {
	//d_decompose<<<1, 1>>>(make_int4(30, 15, 33, 0));
	//d_queryBlock<<<1, 1>>>(make_int4(15, 20, 25, 0), make_int4 (15, 20, 25, 0));	//32, 48, 64, 0

	//d_divideBlock<<<1, 1>>>(30);

	//printf("volumeSize: %d %d %d\n", volumeSize.width, volumeSize.height, volumeSize.depth);
	d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
			brightness, transferOffset, transferScale, queryMethod,
			make_int3(volumeSize.width, volumeSize.height, volumeSize.depth),
			flexBlockSurfObj);
}

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) {
	checkCudaErrors(
			cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
