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

__global__ void ConvLayer_Forward_cu_test(ConvNode *node, double* filters, LayerSize filterSize, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad, int filterCount)
{
	for (int d2 = 0; d2 < filterCount; d2++)
	{
		for (int y = 0; y < layerSize.height; y++)
		{
			for (int x = 0; x < layerSize.width; x++)
			{
				int posx = x - pad;
				int posy = y - pad;
				double val = 0;
				double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * d2);

				for (int filterPosy = 0; filterPosy < filterSize.height; filterPosy++)
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

				val += node[d2].bias;

				output[((y * layerSize.width) + x) * layerSize.depth + d2] = val;
			}
		}
	}
}

__global__ void ConvLayer_Backward_cu_test(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	for (int d2 = 0; d2<filterCount; d2++)
	{
		for (int y = 0; y<layerSize.height; y++)
		{
			for (int x = 0; x<layerSize.width; x++)
			{
				int posx = x - pad;
				int posy = y - pad;
				double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * d2);
				double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * d2);
				double gradient = nextLayerOutput[((layerSize.width * y) + x) * nextLayerSize.depth + d2];

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
								int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
								int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

								backFilter[index2] += previousLayerOutput[index1] * gradient;
								output[index1] += filter[index2] * gradient;
							}
						}
					}
				}

				node[d2].bias += gradient * learnRate;
			}
		}
	}
}

__global__ void ConvLayer_Update_Backward_filter_cu_test(double* filters, double* backFilters, LayerSize filterSize, double learnRate, int filterCount)
{
	int size = filterSize.width * filterSize.height * filterSize.depth;

	for (int i = 0; i < filterCount; i++)
	{
		double* filter = filters + (size * i);
		double* backFilter = backFilters + (size * i);

		for (int index = 0; index < size; index++)
		{
			filter[index] += backFilter[index] * learnRate;
		}
	}
}

void ConvLayer_ForwardReference(ConvNode *node, double* filters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad)
{
	ConvLayer_Forward_cu_test << <1, 1 >> >(node, filters, filterSize, layerSize, previousLayerSize, previousLayerOutput, output, pad, filterCount);

	LayerSynchronize();
}

void ConvLayer_BackwardReference(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	ConvLayer_Backward_cu_test << <1, 1 >> >(node, filters, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);

	LayerSynchronize();

	ConvLayer_Update_Backward_filter_cu_test << <1, 1 >> >(filters, backFilters, filterSize, learnRate, filterCount);

	LayerSynchronize();
}