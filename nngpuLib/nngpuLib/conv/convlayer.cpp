#include <algorithm>
#include <iostream>
#include "convlayerconfig.h"
#include "convlayer.h"
#include "layersize.h"
#include "layerexception.h"
#include <cassert>
#include "cuda_runtime.h"
#include "layer.h"

extern void ConvLayer_Forward(ConvNode *node, double* filters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad);
extern void ConvLayer_Backward(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate);

ConvLayer::ConvLayer(ConvLayerConfig* config, INNetworkLayer* previousLayer)
{
	pad = config->GetPad();
	stride = config->GetStride();

	layerWidth = (int)std::floor((double)(previousLayer->GetWidth() + pad * 2 - config->GetFilterWidth()) / (double)(stride + 1));
	layerHeight = (int)std::floor((double)(previousLayer->GetHeight() + pad * 2 - config->GetFilterHeight()) / (double)(stride + 1));
	layerDepth = config->GetFilterCount();

	filterWidth = config->GetFilterWidth();
	filterHeight = config->GetFilterHeight();
	filterDepth = config->GetFilterDepth();
	filterSize = filterWidth * filterHeight;
	filterCount = config->GetFilterCount();

	backwardCount = previousLayer->GetForwardNodeCount();
	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Convolution,
		forwardCount,
		backwardCount,
		nodeCount,
		true);

	ConvNode* nodes = nodeHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		nodes->bias = 0;
		nodes++;
	}

	filterHostMem = std::unique_ptr<double>(new double[filterSize * filterDepth * filterCount]);
	std::fill_n(filterHostMem.get(), filterSize * filterDepth * filterCount, (double)0.5);
	if (cudaMalloc((void**)&filterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(filterDeviceMem, filterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}

	backFilterHostMem = std::unique_ptr<double>(new double[filterSize * filterDepth * filterCount]);
	std::fill_n(filterHostMem.get(), filterSize * filterDepth * filterCount, (double)0.0);
	if (cudaMalloc((void**)&backFilterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(backFilterDeviceMem, backFilterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}
}

void ConvLayer::Dispose()
{
	Layer::Dispose();
}

void ConvLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for ConvLayer layer");
}

void ConvLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	ConvLayer_Forward(
		nodeDeviceMem, 
		filterDeviceMem, 
		LayerSize(filterWidth, filterHeight, filterDepth), 
		filterCount,
		LayerSize(layerWidth, layerHeight, layerDepth),
		LayerSize(previousLayer->GetWidth(), previousLayer->GetHeight(), previousLayer->GetDepth()),
		previousLayer->GetForwardDeviceMem(), 
		forwardDeviceMem, 
		pad);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(filterHostMem.get(), filterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}
}

void ConvLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for ConvLayer layer");
}

void ConvLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	std::fill_n(backFilterHostMem.get(), filterSize * filterDepth * filterCount, (double)0.0);

	if (cudaMemcpy(backFilterDeviceMem, backFilterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}

	ConvLayer_Backward(
		nodeDeviceMem,
		filterDeviceMem,
		backFilterDeviceMem,
		LayerSize(filterWidth, filterHeight, filterDepth),
		filterCount,
		LayerSize(layerWidth, layerHeight, layerDepth),
		LayerSize(nextLayer->GetWidth(), nextLayer->GetHeight(), nextLayer->GetDepth()),
		previousLayer->GetForwardDeviceMem(),
		nextLayer->GetBackwardDeviceMem(),
		backwardDeviceMem,
		pad,
		learnRate);

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
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
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
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
	return backwardCount;
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
	layerData->width = backwardCount;
	layerData->height = 1;
	layerData->depth = 1;
	layerData->data = GetBackwardHostMem(true);
	layerData++;

	double* filter = filterHostMem.get();
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


void ConvLayer::DebugPrint()
{
	std::cout << "conv layer:\r\n";

	std::cout << "back filters:\r\n";
	double* backFilters = backFilterHostMem.get();
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

}