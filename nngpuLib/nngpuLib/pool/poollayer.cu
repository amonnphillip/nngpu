#pragma once
#include <stdio.h>
#include <stdexcept>

#include "poollayer.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


__global__ void PoolLayer_Forward_cu(double *previousLayerForward, double *out, int* backwardData, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth)
{
	int index = blockIdx.x + (blockIdx.y * width) + (blockIdx.z * width * height);
	for (int y = 0; y < stride; y++)
	{
		for (int x = 0; x < stride; x++)
		{
			int previousLayerIndex = x + (blockIdx.x * stride) + (((blockIdx.y * stride) + y) * previousLayerWidth) + (blockIdx.z * previousLayerWidth * previousLayerHeight);
			double val = previousLayerForward[previousLayerIndex];
			if (val > out[index])
			{
				out[index] = val;
				backwardData[index] = previousLayerIndex;
			}
		}
	}
}

__global__ void PoolLayer_Backward_cu(double* nextlayerBackward, double *out, int* backwardData)
{
	int index = backwardData[blockIdx.x];
	out[index] += nextlayerBackward[blockIdx.x];
}

void PoolLayer_Forward(double *previousLayerForward, double *output, int* backwardData, int nodeCount, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth)
{
	// TODO: For simplicity just use a simple block calculation
	dim3 blocks(width, height, depth);

	// TODO: For simplicity just use one thread for now!
	PoolLayer_Forward_cu <<<blocks, 1 >>>(previousLayerForward, output, backwardData, width, height, depth, stride, previousLayerWidth, previousLayerHeight, previousLayerDepth);

	LayerSynchronize();
}

void PoolLayer_Backward(double* nextlayerBackward, double *output, int* backwardData, int nodeCount)
{
	PoolLayer_Backward_cu <<<nodeCount, 1 >>>(nextlayerBackward, output, backwardData);

	LayerSynchronize();
}