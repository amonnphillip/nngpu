
#pragma once

#include <stdio.h>
#include <stdexcept>

#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <device_double_functions.h>
#include <math_constants.h>
#include <cuda_runtime_api.h>

#include "softmaxlayer.h"


__global__ void SoftmaxLayer_Forward_cu(double *previousLayerForward, double *out, double* expDeviceMem, int nodeCount)
{
	double max = -__longlong_as_double((unsigned long long)0x7ff0000000000000);
	for (int index = 0; index < nodeCount; index++)
	{
		if (previousLayerForward[index] > max)
		{
			max = previousLayerForward[index];
		}
	}

	double sum = 0;
	for (int index = 0; index < nodeCount; index++) {
		expDeviceMem[index] = exp(previousLayerForward[index] - max);
		sum += expDeviceMem[index];
	}

	for (int index = 0; index < nodeCount; index++) {
		out[index] = expDeviceMem[index] / sum;
	}
}

__global__ void SoftmaxLayer_Backward_cu(double* nextlayerBackward, double* out, double* formward, int nodeCount)
{
	for (int index = 0;index < nodeCount;index++) {
		out[index] = -(formward[index] - nextlayerBackward[index]);
	}
}

void SoftmaxLayer_Forward(double *previousLayerForward, double* output, double* expDeviceMem, int nodeCount)
{
	SoftmaxLayer_Forward_cu <<<1, 1 >>>(previousLayerForward, output, expDeviceMem, nodeCount);

	LayerSynchronize();
}

extern void SoftmaxLayer_Backward(double* nextlayerBackward, double* output, double* forward, int nodeCount)
{
	SoftmaxLayer_Backward_cu <<<1, 1 >>>(nextlayerBackward, output, forward, nodeCount);

	LayerSynchronize();
}