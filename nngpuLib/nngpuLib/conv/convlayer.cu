#pragma once
#include <stdio.h>
#include <stdexcept>

#include "convlayer.h"
#include "layersize.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


__global__ void ConvLayer_Forward_cu(ConvNode *node, double* filters, LayerSize filterSize, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad)
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

					if (index1 > filterSize.width * filterSize.height * filterSize.depth)
					{
						val = 0;
					}

					if (index2 > previousLayerSize.width * previousLayerSize.height * previousLayerSize.depth)
					{
						val = 0;
					}
					val += filter[index1] * previousLayerOutput[index2];
				}
			}
		}
	}

	val += node[blockIdx.z].bias;

	output[((blockIdx.y * layerSize.width) + blockIdx.x) * layerSize.depth + blockIdx.z] = val;
}

__global__ void ConvLayer_Backward_update_output_cu(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	// TODO: ASSUMING PAD OF 1!!

	int d = threadIdx.x;

	unsigned int index1 = ((layerSize.width * blockIdx.y) + blockIdx.x) * previousLayerSize.depth + d;

	int fpxStart = -pad;
	int filterStartPosx = filterSize.width - 1;
	if ((int)blockIdx.x - pad < 0)
	{
		fpxStart = 0;
		filterStartPosx = filterSize.width - pad - 1;
	}

	int fpyStart = -pad;
	int filterStartPosy = filterSize.height - 1;
	if ((int)blockIdx.y - pad < 0)
	{
		fpyStart = 0;
		filterStartPosy = filterSize.height - pad - 1;
	}

	int filterEndPosx = 0;
	if ((int)blockIdx.x + filterSize.width - pad > layerSize.width)
	{
		filterEndPosx = (int)blockIdx.x + filterSize.width - pad - layerSize.width;
	}

	int filterEndPosy = 0;
	if ((int)blockIdx.y + filterSize.height - pad > layerSize.height)
	{
		filterEndPosy = (int)blockIdx.y + filterSize.height - pad - layerSize.height;
	}

	for (int filterIndex = 0; filterIndex < filterCount; filterIndex++)
	{
		double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * filterIndex);


		for (int fpy = fpyStart, int filterPosy = filterStartPosy; filterPosy >= filterEndPosy; fpy++, filterPosy--)
		{
			for (int fpx = fpxStart, int filterPosx = filterStartPosx; filterPosx >= filterEndPosx; fpx++, filterPosx--)
			{
				double gradient = nextLayerOutput[((layerSize.width * (blockIdx.y + fpy)) + (blockIdx.x + fpx)) * nextLayerSize.depth + filterIndex];
				int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

				output[index1] += filter[index2] * gradient;
			}
		}
	}
}

__global__ void ConvLayer_Backward_update_bias_cu(ConvNode *node, LayerSize layerSize, LayerSize nextLayerSize, double *nextLayerOutput, double learnRate)
{
	for (int y = 0; y < layerSize.height; y++)
	{
		for (int x = 0; x < layerSize.width; x++)
		{
			double gradient = nextLayerOutput[((layerSize.width * y) + x) * nextLayerSize.depth + blockIdx.x];

			node[blockIdx.x].bias += gradient * learnRate;
		}
	}
}

__global__ void ConvLayer_Backward_back_filters_cu2(double* backFilters, LayerSize filterSize, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, LayerSize nextLayerSize, double *nextLayerOutput, int pad, int* backFilterLookUp)
{
	// do each pixel in each filter!
	// 
	// blockIdx.x = filter x
	// blockIdx.y = filter y
	// blockIdx.z = filter index (count)
	// threadIdx.x = filterSize.depth

	
	double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);

	int index2 = ((filterSize.width * blockIdx.y) + blockIdx.x) * filterSize.depth;
	int* lookUp = backFilterLookUp + (((filterSize.width * blockIdx.y) + blockIdx.x) * layerSize.width * layerSize.height * 2);
	index2 += threadIdx.x;

	for (int y = 0; y < layerSize.height; y++)
	{
		for (int x = 0; x < layerSize.width; x++)
		{
			int index1 = *lookUp;
			lookUp++;
			int gradIndex = *lookUp;
			lookUp++;

			if (index1 >= 0)
			{
				gradIndex += blockIdx.z;

				double gradient = nextLayerOutput[gradIndex];
				index1 += threadIdx.x;

				backFilter[index2] += previousLayerOutput[index1] * gradient;
			}
		}
	}
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

	LayerSynchronize();
}

void ConvLayer_Backward(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate, int* backFilterLookUp, int backFilterLookUpSize)
{
	// TODO: I ASSUME THE PAD IS 1!!

	dim3 bffblocks(filterSize.width, filterSize.height, filterCount);
	ConvLayer_Backward_back_filters_cu2 <<<bffblocks, filterSize.depth>>>(backFilters, filterSize, layerSize, previousLayerSize, previousLayerOutput, nextLayerSize, nextLayerOutput, pad, backFilterLookUp);

	LayerSynchronize();

	dim3 bblocks(layerSize.width, layerSize.height, 1);
	ConvLayer_Backward_update_output_cu << <bblocks, filterSize.depth >> >(node, filters, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);

	LayerSynchronize();

	ConvLayer_Update_Backward_filter_cu <<<filterCount, 1 >>>(filters, backFilters, filterSize, learnRate);

	LayerSynchronize();

	ConvLayer_Backward_update_bias_cu<<<filterCount, 1>>>(node, layerSize, nextLayerSize, nextLayerOutput, learnRate);

	LayerSynchronize();
}