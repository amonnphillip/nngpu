#pragma once
#include <stdio.h>
#include <stdexcept>

#include "fullyconnectedlayer.h"
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void FullyConnectedLayer_Forward_cu(FullyConnectedNode *node, double* weights, int weightCount, double *in, double *out)
{
	double* weightBlock = weights + (blockIdx.x * weightCount);
	double val = 0;
	for (int i = 0; i < weightCount; i++)
	{
		val += *weightBlock * in[i];
		if (blockIdx.x == 0)
		{
			//printf("val: %f, i: %i, in[i]: %f, *weightBlock: %f\n", val, i, in[i], *weightBlock);
		}
		weightBlock++;
	}

	out[blockIdx.x] = val + node->bias;
	if (blockIdx.x == 0)
	{
		//printf("\n");
	}
}

__global__ void FullyConnectedLayer_Backward_cu_p1(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate, int nodeCount)
{
	/*
		double error = nextlayerBackward[blockIdx.x];
	double* weightBlock = weights + (blockIdx.x * weightCount);
	for (int i = 0; i < weightCount; i++)
	{
		*out += *weightBlock * error;
		out++;
		*weightBlock += previousLayerForward[i] * error * learnRate;
		weightBlock++;
	}

	node[blockIdx.x].bias += error * learnRate;
	*/

	out += blockIdx.x;
	//for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
	//{
		double error = nextlayerBackward[blockIdx.y];
		double* weightBlock = weights + (blockIdx.y * weightCount) + blockIdx.x;
		*out += *weightBlock * error;
		//*out += 1;
		//out++;
		//*weightBlock += previousLayerForward[i] * error * learnRate;
		//weightBlock++;
	//}

	//node[blockIdx.x].bias += error * learnRate;
}

__global__ void FullyConnectedLayer_Backward_cu_p2(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate, int nodeCount)
{
	/*
	double error = nextlayerBackward[blockIdx.x];
	double* weightBlock = weights + (blockIdx.x * weightCount);
	for (int i = 0; i < weightCount; i++)
	{
	*out += *weightBlock * error;
	out++;
	*weightBlock += previousLayerForward[i] * error * learnRate;
	weightBlock++;
	}

	node[blockIdx.x].bias += error * learnRate;
	*/
	//for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
	//{
		double* weightBlock = weights + (blockIdx.y * weightCount) + blockIdx.x;
		double error = nextlayerBackward[blockIdx.y];
		*weightBlock += previousLayerForward[blockIdx.x] * error * learnRate;
	//}

	//node[blockIdx.x].bias += error * learnRate;
}

__global__ void FullyConnectedLayer_Backward_cu_p3(FullyConnectedNode *node, double* nextlayerBackward, double learnRate)
{
	double error = nextlayerBackward[blockIdx.x];
	node[blockIdx.x].bias += error * learnRate;
}

__global__ void FullyConnectedLayer_Backward_cu(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate)
{
	double error = nextlayerBackward[blockIdx.x];
	double* weightBlock = weights + (blockIdx.x * weightCount);
	for (int i = 0; i < weightCount; i++)
	{
		*out += *weightBlock * error;
		out++;
		*weightBlock += previousLayerForward[i] * error * learnRate;
		weightBlock++;
	}

	node[blockIdx.x].bias += error * learnRate;
}

__global__ void FullyConnectedLayer_Backward_cu_2(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate, int nodeCount)
{
	for (int x = 0; x < nodeCount; x++)
	{
		double error = nextlayerBackward[x];
		double* weightBlock = weights + (x * weightCount);
		for (int i = 0; i < weightCount; i++)
		{
			out[i] += *weightBlock * error;
			*weightBlock += previousLayerForward[i] * error * learnRate;
			weightBlock++;
		}

		node[x].bias += error * learnRate;
	}
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
	//FullyConnectedLayer_Backward_cu <<<nodeCount, 1 >>>(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate);
	//FullyConnectedLayer_Backward_cu_2 << <1, 1 >> >(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate, nodeCount);
	FullyConnectedLayer_Backward_cu_p1 << <weightCount, nodeCount >> >(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate, nodeCount);
	FullyConnectedLayer_Backward_cu_p2 << <weightCount, nodeCount >> >(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate, nodeCount);
	FullyConnectedLayer_Backward_cu_p3 << <nodeCount, 1 >> >(node, nextlayerBackward, learnRate);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}