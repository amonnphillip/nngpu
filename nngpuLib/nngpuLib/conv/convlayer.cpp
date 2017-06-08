#include <algorithm>
#include <iostream>
#include <cassert>
#include "convlayerconfig.h"
#include "convlayer.h"
#include "layersize.h"
#include "layerexception.h"
#include "cuda_runtime.h"
#include "layer.h"

extern void ConvLayer_Forward(ConvNode *node, double* filters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad);
extern void ConvLayer_Backward(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate, int* backFilterLookUp, int backFilterLookUpSize);

ConvLayer::ConvLayer(ConvLayerConfig* config, INNetworkLayer* previousLayer)
{
	pad = config->GetPad();
	stride = config->GetStride();

	layerWidth = (int)std::floor(((double)(previousLayer->GetWidth() + pad * 2 - config->GetFilterWidth()) / (double)stride) + 1);
	layerHeight = (int)std::floor(((double)(previousLayer->GetHeight() + pad * 2 - config->GetFilterHeight()) / (double)stride) + 1);
	layerDepth = config->GetFilterCount();

	filterWidth = config->GetFilterWidth();
	filterHeight = config->GetFilterHeight();
	filterDepth = previousLayer->GetForwardDepth();
	filterSize = filterWidth * filterHeight;
	filterCount = config->GetFilterCount();

	backwardWidth = previousLayer->GetForwardWidth();
	backwardHeight = previousLayer->GetForwardHeight();
	backwardDepth = previousLayer->GetForwardDepth();

	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Convolution,
		forwardCount,
		backwardWidth * backwardHeight * backwardDepth,
		nodeCount,
		true);

	ConvNode* nodes = nodeHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		nodes->bias = 0;
		nodes++;
	}

	filterHostMem = std::unique_ptr<double>(new double[filterSize * filterDepth * filterCount]);
	if (filterHostMem.get() == nullptr)
	{
		throw std::bad_alloc();
	}

	double filterValue = 1.0 / (double)(filterSize * filterDepth);
	std::fill_n(filterHostMem.get(), filterSize * filterDepth * filterCount, (double)filterValue);
	if (cudaMalloc((void**)&filterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(filterDeviceMem, filterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}

	backFilterHostMem = std::unique_ptr<double>(new double[filterSize * filterDepth * filterCount]);
	if (backFilterHostMem.get() == nullptr)
	{
		throw std::bad_alloc();
	}

	std::fill_n(backFilterHostMem.get(), filterSize * filterDepth * filterCount, (double)0.0);
	if (cudaMalloc((void**)&backFilterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(backFilterDeviceMem, backFilterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}
}

void ConvLayer::Dispose()
{
	if (filterDeviceMem != nullptr)
	{
		if (cudaFree(filterDeviceMem) != cudaError::cudaSuccess)
		{
			throw std::bad_alloc();
		}
		filterDeviceMem = nullptr;
	}

	if (backFilterDeviceMem != nullptr)
	{
		if (cudaFree(backFilterDeviceMem) != cudaError::cudaSuccess)
		{
			throw std::bad_alloc();
		}
		backFilterDeviceMem = nullptr;
	}

	if (backFilterLookUpDeviceMem != nullptr)
	{
		if (cudaFree(backFilterLookUpDeviceMem) != cudaError::cudaSuccess)
		{
			throw std::bad_alloc();
		}
		backFilterLookUpDeviceMem = nullptr;
	}

	Layer::Dispose();
}

void ConvLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for ConvLayer layer");
}

void ConvLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), forwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}

	layerPerf.Start(layerWidth * layerHeight * layerDepth);
	ConvLayer_Forward(
		nodeDeviceMem, 
		filterDeviceMem, 
		LayerSize(filterWidth, filterHeight, filterDepth), 
		filterCount,
		LayerSize(layerWidth, layerHeight, layerDepth),
		LayerSize(previousLayer->GetForwardWidth(), previousLayer->GetForwardHeight(), previousLayer->GetForwardDepth()),
		previousLayer->GetForwardDeviceMem(), 
		forwardDeviceMem, 
		pad);
	layerPerf.Stop();
		
/*
	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(filterHostMem.get(), filterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}*/



#ifdef _UNITTEST
	//DebugPrint();
#endif
}

void ConvLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for ConvLayer layer");
}

void ConvLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	if (backFilterLookUpHostMem.get() == nullptr)
	{
		ComputeBackFilterLookUp(previousLayer, nextLayer);
	}

	std::fill_n(backwardHostMem.get(), GetBackwardNodeCount(), (double)0.0);
	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), GetBackwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}

	std::fill_n(backFilterHostMem.get(), filterSize * filterDepth * filterCount, (double)0.0);
	if (cudaMemcpy(backFilterDeviceMem, backFilterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}

	//layerPerf.Start(layerWidth * layerHeight * layerDepth);
	ConvLayer_Backward(
		nodeDeviceMem,
		filterDeviceMem,
		backFilterDeviceMem,
		LayerSize(filterWidth, filterHeight, filterDepth),
		filterCount,
		LayerSize(layerWidth, layerHeight, layerDepth),
		LayerSize(previousLayer->GetForwardWidth(), previousLayer->GetForwardHeight(), previousLayer->GetForwardDepth()),
		LayerSize(nextLayer->GetBackwardWidth(), nextLayer->GetBackwardHeight(), nextLayer->GetBackwardDepth()),
		previousLayer->GetForwardDeviceMem(),
		nextLayer->GetBackwardDeviceMem(),
		backwardDeviceMem,
		pad,
		learnRate,
		backFilterLookUpDeviceMem,
		backFilterLookupSize);
	//layerPerf.Stop();
	/*
	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(nodeHostMem.get(), nodeDeviceMem, nodeCount * sizeof(ConvNode), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backFilterHostMem.get(), backFilterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}
*/

#ifdef _UNITTEST
	//DebugPrint();
#endif
}

double* ConvLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
		}
	}

	return forwardHostMem.get();
}

double* ConvLayer::GetBackwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
}

double* ConvLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* ConvLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int ConvLayer::GetForwardNodeCount()
{
	return forwardCount;
}

int ConvLayer::GetForwardWidth()
{
	return layerWidth;
}

int ConvLayer::GetForwardHeight()
{
	return layerHeight;
}

int ConvLayer::GetForwardDepth()
{
	return layerDepth;
}

int ConvLayer::GetBackwardNodeCount()
{
	return backwardWidth * backwardHeight * backwardDepth;
}

int ConvLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int ConvLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int ConvLayer::GetBackwardDepth()
{
	return backwardDepth;
}

int ConvLayer::GetWidth()
{
	return layerWidth;
}

int ConvLayer::GetHeight()
{
	return layerHeight;
}

int ConvLayer::GetDepth()
{
	return layerDepth;
}

LayerType ConvLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void ConvLayer::GetLayerData(LayerDataList& layerDataList)
{
	LayerData* layerData = new LayerData[2 + (filterDepth * filterCount)];

	layerDataList.layerDataCount = 2 + (filterDepth * filterCount);
	layerDataList.layerType = LayerType::Convolution;
	layerDataList.layerData = layerData;

	layerData->type = LayerDataType::Forward;
	layerData->width = GetForwardWidth();
	layerData->height = GetForwardHeight();
	layerData->depth = GetForwardDepth();
	layerData->data = GetForwardHostMem(true);
	layerData++;

	layerData->type = LayerDataType::Backward;
	layerData->width = GetBackwardNodeCount();
	layerData->height = 1;
	layerData->depth = 1;
	layerData->data = GetBackwardHostMem(true);
	layerData++;

	double* filter = GetFilterHostMem(true);
	for (int depthIndex = 0; depthIndex < filterDepth; depthIndex++)
	{
		for (int countIndex = 0; countIndex < filterCount; countIndex++)
		{
			layerData->type = LayerDataType::ConvFilter;
			layerData->width = filterWidth;
			layerData->height = filterHeight;
			layerData->depth = 1;
			layerData->data = filter;

			layerData++;
			filter += filterSize;
		}
	}
}

double* ConvLayer::GetFilterHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(filterHostMem.get(), filterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
		}
	}

	return filterHostMem.get();
}

int ConvLayer::GetFilterMemNodeCount()
{
	return filterSize * filterDepth * filterCount;
}

double* ConvLayer::GetBackFilterHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backFilterHostMem.get(), backFilterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
		}
	}

	return backFilterHostMem.get();
}

int ConvLayer::GetBackFilterMemNodeCount()
{
	return filterSize * filterDepth * filterCount;
}

void ConvLayer::ComputeBackFilterLookUp(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	LayerSize layerSize = LayerSize(layerWidth, layerHeight, layerDepth);
	LayerSize filterSize = LayerSize(filterWidth, filterHeight, filterDepth);
	LayerSize previousLayerSize = LayerSize(previousLayer->GetForwardWidth(), previousLayer->GetForwardHeight(), previousLayer->GetForwardDepth());
	LayerSize nextLayerSize = LayerSize(nextLayer->GetBackwardWidth(), nextLayer->GetBackwardHeight(), nextLayer->GetBackwardDepth());

	int lookupSize = layerSize.width * layerSize.height * filterSize.width * filterSize.height;
	int maxHits = layerSize.width * layerSize.height;

	backFilterLookupSize = lookupSize;
	backFilterLookUpHostMem = std::unique_ptr<int>(new int[lookupSize * 2]);
	std::fill_n(backFilterLookUpHostMem.get(), lookupSize * 2, (int)-1);

	int* hitCount = new int[filterSize.width * filterSize.height];
	std::fill_n(hitCount, filterSize.width * filterSize.height, (int)0);

	int* backFilterLookUp = backFilterLookUpHostMem.get();

	for (int y = 0; y<layerSize.height; y++)
	{
		for (int x = 0; x<layerSize.width; x++)
		{
			int posx = x - pad;
			int posy = y - pad;

			int gradIndex = ((layerSize.width * y) + x) * nextLayerSize.depth;

			for (int filterPosy = 0; filterPosy < filterSize.height; filterPosy++)
			{
				for (int filterPosx = 0; filterPosx < filterSize.width; filterPosx++)
				{
					if (filterPosy + posy >= 0 &&
						filterPosy + posy < layerSize.height &&
						filterPosx + posx >= 0 &&
						filterPosx + posx < layerSize.width)
					{

						int index1 = ((layerSize.width * (filterPosy + posy)) + filterPosx + posx) * previousLayerSize.depth;
						int index2 = ((filterSize.width * filterPosy) + filterPosx);

						backFilterLookUp[(index2 * maxHits * 2) + (hitCount[index2] * 2)] = index1;
						backFilterLookUp[(index2 * maxHits * 2) + (hitCount[index2] * 2) + 1] = gradIndex;
						hitCount[index2] ++;
					}
				}
			}
		}
	}

	if (cudaMalloc((void**)&backFilterLookUpDeviceMem, backFilterLookupSize * sizeof(int) * 2) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(backFilterLookUpDeviceMem, backFilterLookUpHostMem.get(), backFilterLookupSize * sizeof(int) * 2, cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}
}

void ConvLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytes)
{
	layerPerf.CalculateAverages(averageTime, averageBytes);
}

void ConvLayer::DebugPrint()
{
	std::cout << "conv layer:\r\n";

	std::cout << "back filters:\r\n";
	double* backFilters = GetBackFilterHostMem(true);
	for (int c = 0; c < filterCount; c++)
	{
		for (int d = 0; d < filterDepth; d++)
		{
			for (int y = 0; y < filterWidth; y++)
			{
				for (int x = 0; x < filterHeight; x++)
				{
					std::cout << *backFilters << " ";
					backFilters++;
				}
				std::cout << "\r\n";
			}
			std::cout << "\r\n";
		}
	}
#if 0
	std::cout << "forward filters:\r\n";
	double* forwardFilters = filterHostMem.get();
	for (int c = 0; c < filterCount; c++)
	{
		for (int d = 0; d < filterDepth; d++)
		{
			for (int y = 0; y < filterWidth; y++)
			{
				for (int x = 0; x < filterHeight; x++)
				{
					std::cout << *forwardFilters << " ";
					forwardFilters++;
				}
				std::cout << "\r\n";
			}
			std::cout << "\r\n";
		}
	}

	std::cout << "forward:\r\n";
	double* forward = forwardHostMem.get();
	for (int d = 0; d < layerDepth; d++)
	{
		for (int y = 0; y < layerWidth; y++)
		{
			for (int x = 0; x < layerHeight; x++)
			{
				std::cout << *forward << " ";
				forward++;
			}
			std::cout << "\r\n";
		}
		std::cout << "\r\n";
	}
#endif
}