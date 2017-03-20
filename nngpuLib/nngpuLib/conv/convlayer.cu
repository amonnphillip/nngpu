#pragma once

#include "convlayer.h"
#include "layersize.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>

__global__ void ConvLayer_Forward_cu(ConvNode *node, double* filters, LayerSize filterSize, LayerSize layerSize, LayerSize previousLayerSize, double *input, double *output, int pad)
{
	int posx = blockIdx.x - pad;
	int posy = blockIdx.y - pad;
	double val = 0;
	double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);

	for (int filterPosy = 0; filterPosy < filterSize.height;filterPosy++)
	{
		for (int filterPosx = 0; filterPosx < filterSize.width; filterPosx++)
		{
			if (filterPosy + posy >= 0 &&
				filterPosy + posy < previousLayerSize.height &&
				filterPosx + posx >= 0 &&
				filterPosx + posx < previousLayerSize.width)
			{
				for (int d = 0; d < filterSize.depth; d++)
				{
					int index1 = ((filterPosy * filterSize.width) + filterPosx) * filterSize.depth + d;
					int index2 = (((posy + filterPosy) * previousLayerSize.width) + posx + filterPosx) * previousLayerSize.depth + d;
					val += filter[index1] * input[index2];
				}
			}
		}
	}

	val += node->bias;
	output[((blockIdx.y * layerSize.width) + blockIdx.x) * layerSize.depth + blockIdx.z] = val;
}

__global__ void ConvLayer_Backward_cu(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, int learnRate)
{
	int posx = blockIdx.x - pad;
	int posy = blockIdx.y - pad;
	double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
	double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
	double gradient = nextLayerOutput[((layerSize.width * blockIdx.y) + blockIdx.x) * nextLayerSize.depth + blockIdx.z];

	for (int filterPosy = 0; filterPosy < filterSize.height; filterPosy++)
	{
		for (int filterPosx = 0; filterPosx < filterSize.width; filterPosx++)
		{
			if (filterPosy + posy >= 0 &&
				filterPosy + posy < layerSize.height &&
				filterPosx + posx >= 0 &&
				filterPosx + posx < layerSize.width)
			{
				for (int d = 0; d < filterSize.depth; d++)
				{
					int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * layerSize.depth + d;
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					backFilter[index2] += previousLayerOutput[index1] * gradient;
					output[index1] += filter[index2] * gradient;
				}
			}
		}
	}

	node->bias += gradient * learnRate;
}

__global__ void ConvLayer_Update_Backward_filter_cu(double* filters, double* backFilters, LayerSize filterSize, double learnRate)
{
	double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.x);
	double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.x);

	int size = filterSize.width * filterSize.height * filterSize.depth;
	for (int index = 0; index < size; index++)
	{
		filter[index] += backFilter[index] * learnRate;
	}
}

void ConvLayer_Forward(ConvNode *node, double* filters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad)
{
	dim3 blocks(layerSize.width, layerSize.height, filterCount);
	ConvLayer_Forward_cu <<<blocks, 1>>>(node, filters, filterSize, layerSize, previousLayerSize, previousLayerOutput, output, pad);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}

void ConvLayer_Backward(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	dim3 blocks(layerSize.width, layerSize.height, filterCount);
	ConvLayer_Backward_cu <<<blocks, 1>>>(node, filters, backFilters, filterSize, filterCount, layerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);

	ConvLayer_Update_Backward_filter_cu <<<filterCount, 1 >>>(filters, backFilters, filterSize, learnRate);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}