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

__global__ void FullyConnectedLayer_Forward_reference_cu(FullyConnectedNode *node, double* weights, int weightCount, double *previousLayerForward, double *out, int nodeCount)
{
	for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
		double val = 0;
		double* weightsList = weights + (weightCount * nodeIndex);
		for (int inputIndex = 0; inputIndex < weightCount; inputIndex++) {
			val += previousLayerForward[inputIndex] * weightsList[inputIndex];
		}

		out[nodeIndex] = val + node[nodeIndex].bias;
	}
}

__global__ void FullyConnectedLayer_Backward_reference_cu(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *out, double learnRate, int nodeCount)
{
	for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
		double gradient = nextlayerBackward[nodeIndex];

		double* weightsList = weights + (weightCount * nodeIndex);
		for (int weightIndex = 0; weightIndex < weightCount; weightIndex++) {
			out[weightIndex] += weightsList[weightIndex] * gradient;
			weightsList[weightIndex] += previousLayerForward[weightIndex] * gradient * learnRate;
		}

		node[nodeIndex].bias += gradient * learnRate;
	}
}

void FullyConnectedLayer_ForwardReference(FullyConnectedNode *node, double* weights, int weightCount, double *previousLayerForward, double *output, int nodeCount)
{
	FullyConnectedLayer_Forward_reference_cu << <1, 1 >> >(node, weights, weightCount, previousLayerForward, output, nodeCount);

	LayerSynchronize();
}

void FullyConnectedLayer_BackwardReference(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate)
{
	FullyConnectedLayer_Backward_reference_cu << <1, 1 >> >(node, weights, weightCount, forward, previousLayerForward, nextlayerBackward, output, learnRate, nodeCount);

	LayerSynchronize();
}