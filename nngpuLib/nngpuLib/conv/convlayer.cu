#pragma once

#include "convlayer.h"
#include "layersize.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdexcept>

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

	//val += node->bias;
	output[((blockIdx.y * layerSize.width) + blockIdx.x) * layerSize.depth + blockIdx.z] = val;
}


__global__ void ConvLayer_Backward_cu(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
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
					int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					backFilter[index2] += previousLayerOutput[index1] * gradient;
					output[index1] += filter[index2] * gradient;
				}
			}
		}
	}

	//node->bias += gradient * learnRate;
}

__global__ void ConvLayer_Backward_update_back_filters_cu(ConvNode *node, double* filters, double* backFilterCollation, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	/*
	blockIdx.xxx is layer size

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
					int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					backFilter[index2] += previousLayerOutput[index1] * gradient;
					output[index1] += filter[index2] * gradient;
				}
			}
		}
	}*/

	int posx = blockIdx.x - pad;
	int posy = blockIdx.y - pad;
	//double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
	unsigned int collationFilterGroupSize = filterSize.width * filterSize.height * filterSize.depth * layerSize.width * layerSize.height;
	double* backFilter = backFilterCollation + (collationFilterGroupSize * blockIdx.z);// *blockIdx.x * blockIdx.y);
	backFilter += (filterSize.width * filterSize.height * filterSize.depth * layerSize.width * blockIdx.y);
	backFilter += (filterSize.width * filterSize.height * filterSize.depth * blockIdx.x);
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
					int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					backFilter[index2] += previousLayerOutput[index1] * gradient;
					//output[index1] += filter[index2] * gradient;
				}
			}
		}

	}
	/*



	for (int d2 = 0; d2<filterCount; d2++)
	{
		//double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * d2);
		double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * d2);

		for (int y = 0; y<layerSize.height; y++)
		{
			for (int x = 0; x<layerSize.width; x++)
			{
				int posx = x - pad;
				int posy = y - pad;
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
								//output[index1] += filter[index2] * gradient;
							}
						}
					}
				}
			}
		}
	}

	blockIdx.x = filterPosx
	blockIdx.y = filterPosy
	blockIdx.z = d

	call for each filter = d2

	SHARED MEMORY?
	threadIdx.x = d2 = shared memory?


	if (blockIdx.y + posy >= 0 &&
		blockIdx.y + posy < layerSize.height &&
		blockIdx.x + posx >= 0 &&
		blockIdx.x + posx < layerSize.width)
	{
		int index1 = ((layerSize.width * (blockIdx.y + posy)) + blockIdx.x + posx) * previousLayerSize.depth + blockIdx.z;
		int index2 = ((filterSize.width * blockIdx.y) + blockIdx.x) * filterSize.depth + blockIdx.z;

		backFilter[index2] += previousLayerOutput[index1] * gradient;
	}


	*/



	//blockIdx.z = filter count
	//	blockIdx.x, y = filter x and y
	//	threadIdx.z = depth... not needed?
	//	threadIdx.x = layer x
	//	threadIdx.y = layer y
/*
	double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
	backFilter += ((filterSize.width * blockIdx.y) + blockIdx.x) * filterSize.depth + threadIdx.z;

	double gradient = nextLayerOutput[((layerSize.width * threadIdx.y) + threadIdx.x) * nextLayerSize.depth + blockIdx.z];
	int posx = threadIdx.x - pad;
	int posy = threadIdx.y - pad;
	if (blockIdx.y + posy >= 0 &&
		blockIdx.y + posy < layerSize.height &&
		blockIdx.x + posx >= 0 &&
		blockIdx.x + posx < layerSize.width)
	{
		for (int d = 0; d < filterSize.depth; d++)
		{
			int index1 = ((layerSize.width * (blockIdx.y + posy)) + blockIdx.x + posx) * previousLayerSize.depth + d;
			*backFilter += previousLayerOutput[index1] * gradient;
		}
	}*/
	/*
	double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);

	for (int y = 0; y < layerSize.height; y++)
	{
		for (int x = 0; x < layerSize.width; x++)
		{
			int posx = x - pad;
			int posy = y - pad;
			double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
			//double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
			double gradient = nextLayerOutput[((layerSize.width * y) + x) * nextLayerSize.depth + blockIdx.z];

			//for (int filterPosy = 0; filterPosy < filterSize.height; filterPosy++)
			//{
				//for (int filterPosx = 0; filterPosx < filterSize.width; filterPosx++)
				//{
					if (blockIdx.y + posy >= 0 &&
						blockIdx.y + posy < layerSize.height &&
						blockIdx.x + posx >= 0 &&
						blockIdx.x + posx < layerSize.width)
					{
						for (int d = 0; d < filterSize.depth; d++)
						{
							int index1 = ((layerSize.width * (blockIdx.y + posy)) + blockIdx.x + posx) * previousLayerSize.depth + d;
							int index2 = ((filterSize.width * blockIdx.y) + blockIdx.x) * filterSize.depth + d;

							backFilter[index2] += previousLayerOutput[index1] * gradient;
							//output[index1] += filter[index2] * gradient;
						}
					}
				//}
			//}
		}
	}
	*/
	//node->bias += gradient * learnRate;
}

__global__ void ConvLayer_Backward_update_back_filters_collate(double* backFilterCollation, double* backFilters, LayerSize filterSize, LayerSize layerSize)
{
	// do each pixel in each filter!
	// 
    // blockIdx.x = filter x
	// blockIdx.y = filter y
	// blockIdx.z = filter index (count)

	// TODO: I ASSUME THE PAD IS 1!!

	unsigned int collationFilterGroupSize = filterSize.width * filterSize.height * filterSize.depth * layerSize.width * layerSize.height;

	double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z) + (blockIdx.y * filterSize.width) + blockIdx.x;
	double* collation = backFilterCollation + (collationFilterGroupSize * blockIdx.z) + (blockIdx.y * filterSize.width) + blockIdx.x;
	for (int y = 0; y < layerSize.height; y++)
	{
		for (int x = 0; x < layerSize.width; x++)
		{
			*backFilter += *collation;
			collation += filterSize.width * filterSize.height * filterSize.depth;
		}
	}
}

__global__ void ConvLayer_Backward_update_output_cu(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	/*
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
					int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					backFilter[index2] += previousLayerOutput[index1] * gradient;
					output[index1] += filter[index2] * gradient;
				}
			}
		}
	}*/

	/*
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
					int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					//backFilter[index2] += previousLayerOutput[index1] * gradient;
					output[index1] += filter[index2] * gradient;
				}
			}
		}
	}*/

	// TODO: ASSUMING PAD OF 1!!

	//int posx = blockIdx.x - pad;
	//int posy = blockIdx.y - pad;
	int d = blockIdx.z;

	//unsigned int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth + d;
	unsigned int index1 = ((layerSize.width * blockIdx.y) + blockIdx.x) * previousLayerSize.depth + d;

	for (int filterIndex = 0; filterIndex < filterCount; filterIndex++)
	{
		double gradient = nextLayerOutput[((layerSize.width * blockIdx.y) + blockIdx.x) * nextLayerSize.depth + filterIndex];

		for (int filterPosy = 0; filterPosy < filterSize.height; filterPosy++)
		{
			for (int filterPosx = 0; filterPosx < filterSize.width; filterPosx++)
			{
				if (blockIdx.x - pad + filterPosx >= 0 &&
					blockIdx.x - pad + filterPosx < layerSize.width &&
					blockIdx.y - pad + filterPosy >= 0 &&
					blockIdx.y - pad + filterPosy < layerSize.height) {
					double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * filterIndex);
					int index2 = ((filterSize.width * filterPosy) + filterPosx) * filterSize.depth + d;

					output[index1] += filter[index2] * gradient;
				}
			}
		}
	}

	/*

	int posx = ((int)blockIdx.x) - pad;
	int posy = ((int)blockIdx.y) - pad;
	double* filter = filters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
	//double* backFilter = backFilters + (filterSize.width * filterSize.height * filterSize.depth * blockIdx.z);
	double gradient = nextLayerOutput[((layerSize.width * blockIdx.y) + blockIdx.x) * nextLayerSize.depth + blockIdx.z];

	//for (int filterPosy = 0; filterPosy < filterSize.height; filterPosy++)
	//{
		//for (int filterPosx = 0; filterPosx < filterSize.width; filterPosx++)
		//{
			if (threadIdx.y + posy >= 0 &&
				threadIdx.y + posy < layerSize.height &&
				threadIdx.x + posx >= 0 &&
				threadIdx.x + posx < layerSize.width)
			{
				for (int d = 0; d < filterSize.depth; d++)
				{
					int index1 = ((layerSize.width * (threadIdx.y + posy)) + threadIdx.x + posx) * previousLayerSize.depth + d;
					int index2 = ((filterSize.width * threadIdx.y) + threadIdx.x) * filterSize.depth + d;

					//backFilter[index2] += previousLayerOutput[index1] * gradient;
					output[index1] += filter[index2] * gradient;

					//__syncthreads();
				}
			}
		//}
	//}*/

	//node->bias += gradient * learnRate;
}

__global__ void ConvLayer_Backward_cu_2(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	for (int d2 = 0;d2<filterCount;d2++)
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

								//backFilter[index2] += previousLayerOutput[index1] * gradient;
								output[index1] += filter[index2] * gradient;
							}
						}
					}
				}
			}
		}
	}



	//node->bias += gradient * learnRate;
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
//	ConvLayer_Forward_cu_2 << <1, 1 >> >(node, filters, filterSize, layerSize, previousLayerSize, previousLayerOutput, output, pad);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}

void ConvLayer_Backward(ConvNode *node, double* filters, double* backFilterCollation, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate)
{
	// TODO: I ASSUME THE PAD IS 1!!

	dim3 blocks(layerSize.width, layerSize.height, filterCount);
	ConvLayer_Backward_update_back_filters_cu <<<blocks, 1>>>(node, filters, backFilterCollation, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);
	//ConvLayer_Backward_cu_2 << <1, 1 >> >(node, filters, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);

	dim3 bfblocks(filterSize.width, filterSize.height, filterCount);
	ConvLayer_Backward_update_back_filters_collate<<<bfblocks , 1>>>(backFilterCollation, backFilters, filterSize, layerSize);


	//dim3 bfblocks(filterSize.width, filterSize.height, filterCount);
	//dim3 bthreads(layerSize.width, layerSize.height, 1);
//	ConvLayer_Backward_update_back_filters_cu <<<bfblocks, bthreads >>>(node, filters, backFilterCollation, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);
	
	dim3 bblocks(layerSize.width, layerSize.height, filterSize.depth);
	//dim3 bthreads2(filterSize.width, filterSize.height, 1);
	ConvLayer_Backward_update_output_cu <<<bblocks, 1 >>>(node, filters, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);


	//ConvLayer_Backward_cu_2 << <1, 1 >> >(node, filters, backFilters, filterSize, filterCount, layerSize, previousLayerSize, nextLayerSize, previousLayerOutput, nextLayerOutput, output, pad, learnRate);
	//ConvLayer_Update_Backward_filter_cu <<<filterCount, 1 >>>(filters, backFilters, filterSize, learnRate);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyconnectedLayer Forward CUDA syncronize returned an error");
	}
}