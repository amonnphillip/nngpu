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
		weightBlock++;
	}

	out[blockIdx.x] = val + node[blockIdx.x].bias;
}

__global__ void FullyConnectedLayer_Backward_cu_p1(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate, int nodeCount)
{
	for (int i = 0; i < nodeCount; i++)
	{
		double error = nextlayerBackward[i];
		double* weightBlock = weights + (i * weightCount) + blockIdx.x;
		out[blockIdx.x] += *weightBlock * error;
	}
}

__global__ void FullyConnectedLayer_Backward_cu_p2(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate, int nodeCount)
{
	double* weightBlock = weights + (blockIdx.y * weightCount) + blockIdx.x;
	double error = nextlayerBackward[blockIdx.y];
	*weightBlock += previousLayerForward[blockIdx.x] * error * learnRate;
}

__global__ void FullyConnectedLayer_Backward_cu_p3(FullyConnectedNode *node, double* nextlayerBackward, double learnRate)
{
	double error = nextlayerBackward[blockIdx.x];
	node[blockIdx.x].bias += error * learnRate;
}

void FullyConnectedLayer_Forward(FullyConnectedNode *node, double* weights, int weightCount, double *input, double *output, int nodeCount)
{
	FullyConnectedLayer_Forward_cu <<<nodeCount, 1 >>>(node, weights, weightCount, input, output);

	LayerSynchronize();
}

void FullyConnectedLayer_Backward(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate)
{
	FullyConnectedLayer_Backward_cu_p1 <<<weightCount, 1 >>>(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate, nodeCount);

	LayerSynchronize();

	dim3 blocks(weightCount, nodeCount, 1);
	FullyConnectedLayer_Backward_cu_p2 <<<blocks, 1 >>>(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate, nodeCount);

	LayerSynchronize();

	FullyConnectedLayer_Backward_cu_p3 <<<nodeCount, 1 >>>(node, nextlayerBackward, learnRate);

	LayerSynchronize();
}