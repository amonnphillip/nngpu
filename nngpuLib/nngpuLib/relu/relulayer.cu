#pragma once
#include <stdio.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include "relulayer.h"

__global__ void ReluLayer_Forward_cu(double *previousLayerForward, double *out)
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

__global__ void ReluLayer_Backward_cu(double *forward, double* nextlayerBackward, double *out, double learnRate)
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

void ReluLayer_Forward(double *previousLayerForward, double *output, int nodeCount)
{
	ReluLayer_Forward_cu <<<nodeCount, 1 >>>(previousLayerForward, output);

	LayerSynchronize();
}

void ReluLayer_Backward(double *forward, double* nextlayerBackward, double *output, int nodeCount, double learnRate)
{
	ReluLayer_Backward_cu <<<nodeCount, 1 >>>(forward, nextlayerBackward, output, learnRate);

	LayerSynchronize();
}