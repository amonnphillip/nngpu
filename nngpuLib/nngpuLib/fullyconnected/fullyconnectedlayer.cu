#pragma once

#include "fullyconnectedlayer.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>

__global__ void FullyConnectedLayer_Forward_cu(FullyConnectedNode *node, double* weights, int weightCount, double *in, double *out)
{
	double* weightBlock = weights + (blockIdx.x * weightCount);
	double val = 0;
	for (int i = 0; i < weightCount; i++)
	{
		val += *weightBlock * in[i];
		weightBlock++;
	}

	out[blockIdx.x] = val + node->bias;
}

__global__ void FullyConnectedLayer_Backward_cu(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate)
{
	double error = nextlayerBackward[blockIdx.x];
	double* weightBlock = weights + (blockIdx.x * weightCount);
	for (int i = 0; i < weightCount; i++)
	{
		out[i] += *weightBlock * error;
		*weightBlock += previousLayerForward[i] * error * learnRate;
		weightBlock++;
	}

	node[blockIdx.x].bias += error * learnRate;
}

void FullyConnectedLayer_Forward(FullyConnectedNode *node, double* weights, int weightCount, double *input, double *output, int nodeCount)
{
	FullyConnectedLayer_Forward_cu <<<nodeCount, 1 >>>(node, weights, weightCount, input, output);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}

void FullyConnectedLayer_Backward(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate)
{
	FullyConnectedLayer_Backward_cu <<<nodeCount, 1 >>>(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}