
#include "relulayer.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>

__global__ void PoolLayer_Forward_reference_cu(double *previousLayerForward, double *out, int* backwardData, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth)
{
	for (int d = 0;d < depth;d++)
	{
		for (int y = 0;y < height;y++)
		{
			for (int x = 0;x < width;x++)
			{
				int index = x + (y * width) + (d * width * height);
				for (int ys = 0;ys < stride;ys++)
				{
					for (int xs = 0;xs < stride;xs++)
					{
						int previousLayerIndex = xs + (x * stride) + (((y * stride) + ys) * previousLayerWidth) + (d * previousLayerWidth * previousLayerHeight);
						double val = previousLayerForward[previousLayerIndex];
						if (val > out[index])
						{
							out[index] = val;
							backwardData[index] = previousLayerIndex;
						}
					}
				}
			}
		}
	}
}

__global__ void PoolLayer_Backward_reference_cu(double* nextlayerBackward, double *out, int* backwardData, int nodeCount)
{
	for (int i = 0;i < nodeCount;i++)
	{
		int index = backwardData[i];
		out[index] += nextlayerBackward[i];
	}
}

void PoolLayer_Forward_reference(double *previousLayerForward, double *output, int* backwardData, int nodeCount, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth)
{
	PoolLayer_Forward_reference_cu << <1, 1 >> >(previousLayerForward, output, backwardData, width, height, depth, stride, previousLayerWidth, previousLayerHeight, previousLayerDepth);

	LayerSynchronize();
}

void PoolLayer_Backward_reference(double* nextlayerBackward, double *output, int* backwardData, int nodeCount)
{
	PoolLayer_Backward_reference_cu << <1, 1 >> >(nextlayerBackward, output, backwardData, nodeCount);

	LayerSynchronize();
}