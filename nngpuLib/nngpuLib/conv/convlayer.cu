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
					/*
					if (blockIdx.x == 1 && blockIdx.y == 0)
					{
						printf("gradient x: %i, y: %i, d2: %i, layerSize.width: %i !\n", blockIdx.x, blockIdx.y, blockIdx.z, layerSize.width);
						printf("filterPosx: %i, filterPosy: %i, posx: %i, posy: %i\n", filterPosx, filterPosy, posx, posy);
						printf("filter[index1]: %f, previousLayerOutput[index2]: %f\n", filter[index1], previousLayerOutput[index2]);
						printf("d: %i\n", d);
						printf("index1: %i\n index2: %i\n", index1, index2);
					}*/

				}
			}
		}
	}

	val += node[blockIdx.z].bias;

	output[((blockIdx.y * layerSize.width) + blockIdx.x) * layerSize.depth + blockIdx.z] = val;
	/*if (blockIdx.x == 1 && blockIdx.y == 0) {
		printf("output[1]: %f\n", output[1]);
	}*/
}

__global__ void ConvLayer_Backward_update_output_cu(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	// TODO: ASSUMING PAD OF 1!!

	//int posx = blockIdx.x - pad;
	//int posy = blockIdx.y - pad;
	int d = threadIdx.x;

	//unsigned int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
	unsigned int index1 = ((layerSize.width * blockIdx.y) + blockIdx.x) * previousLayerSize.depth + d;
	//unsigned int filterStartX = filterSize.width - pad - 1 + blockIdx.x;
	//unsigned int filterStartY = filterSize.height - pad - 1 + blockIdx.y;

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
	/*
	if (blockIdx.x == 11 && blockIdx.y == 0)
	{
	printf("filterStartPosx: %i\n", filterStartPosx);
	printf("filterStartPosy: %i\n", filterStartPosy);

	printf("filterEndPosx: %i\n", filterEndPosx);
	printf("filterEndPosy: %i\n", filterEndPosy);

	printf("?? (int)blockIdx.x + filterSize.width: %i\n", (int)blockIdx.x + filterSize.width);
	printf("?? (int)blockIdx.y + filterSize.height: %i\n", (int)blockIdx.y + filterSize.height);
	printf("?? layerSize.height: %i\n", layerSize.height);
	}*/


	for (int filterIndex = 0; filterIndex < filterCount; filterIndex++)
	{
		double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * filterIndex);


		for (int fpy = fpyStart, int filterPosy = filterStartPosy; filterPosy >= filterEndPosy; fpy++, filterPosy--)
		{
			for (int fpx = fpxStart, int filterPosx = filterStartPosx; filterPosx >= filterEndPosx; fpx++, filterPosx--)
			{

				//int fpx = filterStartX + filterPosx - blockIdx.x + filterSize.width - 1;
				//int fpy = filterStartY + filterPosy - blockIdx.y + filterSize.height - 1;
				double gradient = nextLayerOutput[((layerSize.width * (blockIdx.y + fpy)) + (blockIdx.x + fpx)) * nextLayerSize.depth + filterIndex];
				/*
				if (blockIdx.x == 11 && blockIdx.y == 0)
				{
				printf("fpx: %i, fpy: %i\n", fpx, fpy);
				printf("gradient x: %i, y: %i nextLayerSize.depth: %i, d2: %i, layerSize.width: %i \n", blockIdx.x + fpx, blockIdx.y + fpy, nextLayerSize.depth, filterIndex, layerSize.width);
				}*/

				int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

				output[index1] += filter[index2] * gradient;
				/*
				if (blockIdx.x == 11 && blockIdx.y == 0)
				{
				printf("gradient: %f\n", gradient);
				printf("filter: %i\n", filterIndex);
				printf("index1: %i\n index2: %i\n", index1, index2);
				}*/

			}
		}
	}
	/*
	if (blockIdx.x == 11 && blockIdx.y == 0)
	{
	printf("index1 %i\n", index1);
	printf("output[index1] %f\n", output[index1]);
	}*/

	//node->bias += gradient * learnRate;
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

//	int ii = 0;

	//	int index2 = ((filterSize.width * blockIdx.x) + blockIdx.y) * filterSize.depth + blockIdx.z;
	//int d = 0;
	//for (int d = 0; d < filterSize.depth; d++)
	{
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

					//double gradient = nextLayerOutput[((layerSize.width * (y - pad)) + (x - pad)) * nextLayerSize.depth + blockIdx.z];
					double gradient = nextLayerOutput[gradIndex];

					//int index1 = ((layerSize.width * (y - pad)) + (x - pad)) * previousLayerSize.depth + threadIdx.x;

					//double gradient = nextLayerOutput[((layerSize.width * y) + x) * nextLayerSize.depth + d2];
					//int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;

					index1 += threadIdx.x;
					/*
					if (index2 == 32 && blockIdx.z == 0)
					{
						if (ii < 4000) {
							printf("iiiii: %i, index1: %i, index2: %i, filterPosx: %i, filterPosy: %i, posx: %i, posy: %i, gradient: %f, d: %i, d2: %i, x: %i, y: %i \n", ii, index1, index2, blockIdx.x, blockIdx.y, x, y, gradient, threadIdx.x, blockIdx.z, x - pad, y - pad);
							ii++;
						}
					}*/

					backFilter[index2] += previousLayerOutput[index1] * gradient;
				}
			}
		}
		/*
		if (index2 == 32 && blockIdx.z == 0)
		{
			printf("backFilters[32] %f\n", backFilters[32]);
		}*/
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