#pragma once

#include "relulayer.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>

__global__ void PoolLayer_Forward_cu(double *previousLayerForward, double *out, int* backwardData, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth)
{
	double outVal = out[blockIdx.x + (blockIdx.y * width) + (blockIdx.z * width * height)];
	int outValIndex = 0;
	for (int y = 0; y < stride; y++)
	{
		for (int x = 0; x < stride; x++)
		{
			int previousLayerIndex = x + (blockIdx.x * stride) + ((y + (blockIdx.y * stride)) * previousLayerWidth) + (blockIdx.z * previousLayerWidth * previousLayerHeight);
			double val = previousLayerForward[previousLayerIndex];
			if (val > outVal)
			{
				outVal = val;
				outValIndex = previousLayerIndex;
			}
		}
	}
	backwardData[blockIdx.x + (blockIdx.y * width) + (blockIdx.z * width * height)] = outValIndex;
	out[blockIdx.x + (blockIdx.y * width) + (blockIdx.z * width * height)] = outVal;
}

__global__ void PoolLayer_Backward_cu(double* nextlayerBackward, double *out, int* backwardData, int count)
{
	for (int index = 0; index < count; index++)
	{
		out[*backwardData] += *nextlayerBackward;
		backwardData++;
		out++;
		nextlayerBackward++;
	}
}

void PoolLayer_Forward(double *previousLayerForward, double *output, int* backwardData, int nodeCount, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth)
{
	// TODO: For simplicity just use a simple block calculation
	dim3 blocks(width, height, depth);

	// TODO: For simplicity just use one thread for now!
	PoolLayer_Forward_cu <<<blocks, 1 >>>(previousLayerForward, output, backwardData, width, height, depth, stride, previousLayerWidth, previousLayerHeight, previousLayerDepth);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer Forward CUDA syncronize returned an error");
	}
}

void PoolLayer_Backward(double* nextlayerBackward, double *output, int* backwardData, int nodeCount)
{
	PoolLayer_Backward_cu <<<nodeCount, 1 >>>(nextlayerBackward, output, backwardData, nodeCount);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer Forward CUDA syncronize returned an error");
	}
}