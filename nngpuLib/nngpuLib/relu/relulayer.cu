#pragma once

#include "relulayer.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>

__global__ void ReluLayer_Forward_cu(ReluNode *node, double *previousLayerForward, double *out)
{
	if (previousLayerForward[blockIdx.x] < 0)
	{
		out[blockIdx.x] = 0;
	}
	else
	{
		out[blockIdx.x] = previousLayerForward[blockIdx.x];
	}
}

__global__ void ReluLayer_Backward_cu(ReluNode *node, double *forward, double* nextlayerBackward, double *out, double learnRate)
{
	if (forward[blockIdx.x] <= 0)
	{
		out[blockIdx.x] = 0;
	}
	else
	{
		out[blockIdx.x] = nextlayerBackward[blockIdx.x];
	}
}

void ReluLayer_Forward(ReluNode *node, double *previousLayerForward, double *output, int nodeCount)
{
	ReluLayer_Forward_cu <<<nodeCount, 1 >>>(node, previousLayerForward, output);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer Forward CUDA syncronize returned an error");
	}
}

void ReluLayer_Backward(ReluNode *node, double *forward, double* nextlayerBackward, double *output, int nodeCount, double learnRate)
{
	ReluLayer_Backward_cu <<<nodeCount, 1 >>>(node, forward, nextlayerBackward, output, learnRate);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer Forward CUDA syncronize returned an error");
	}
}