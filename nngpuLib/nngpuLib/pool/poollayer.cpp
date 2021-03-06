#include <cassert>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#include "poollayer.h"
#include "layerexception.h"
#include "testutils.h"

extern void PoolLayer_Forward(double *previousLayerForward, double *output, int* backData, int nodeCount, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth);
extern void PoolLayer_Backward(double* nextlayerBackward, double *output, int* backwardData, int nodeCount);

PoolLayer::PoolLayer(PoolLayerConfig* config, INNetworkLayer* previousLayer)
{
	layerWidth = (previousLayer->GetForwardWidth() - config->GetSpatialExtent()) / config->GetStride() + 1;
	layerHeight = (previousLayer->GetForwardHeight() - config->GetSpatialExtent()) / config->GetStride() + 1;
	layerDepth = previousLayer->GetForwardDepth();
	spatiallExtent = config->GetSpatialExtent();
	stride = config->GetStride();

	backwardWidth = previousLayer->GetForwardWidth();
	backwardHeight = previousLayer->GetForwardHeight();
	backwardDepth = previousLayer->GetForwardDepth();
	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Pool,
		forwardCount,
		backwardWidth * backwardHeight * backwardDepth,
		nodeCount,
		true);

	backDataHostMem = std::unique_ptr<int>(new int[nodeCount]);
	if (backDataHostMem.get() == nullptr)
	{
		throw std::bad_alloc();
	}

	int* backData = backDataHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*backData = 0;
		backData++;
	}

	if (cudaMalloc((void**)&backDataDeviceMem, nodeCount * sizeof(int)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(backDataDeviceMem, backDataHostMem.get(), nodeCount * sizeof(int), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer cudaMemcpy returned an error");
	}
}

void PoolLayer::Dispose()
{
	Layer::Dispose();
}

void PoolLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for PoolLayer");
}

void PoolLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	double negativeInfinity = -std::numeric_limits<double>::infinity();
	std::fill_n(GetForwardHostMem(false), GetForwardNodeCount(), negativeInfinity);
	std::fill_n(GetBackDataHostMem(false), GetBackDataNodeCount(), (int)0);

	if (cudaMemcpy(forwardDeviceMem, GetForwardHostMem(false), GetForwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backDataDeviceMem, GetBackDataHostMem(false), GetBackDataNodeCount() * sizeof(int), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	PoolLayer_Forward(previousLayer->GetForwardDeviceMem(), forwardDeviceMem, backDataDeviceMem, nodeCount, layerWidth, layerHeight, layerDepth, stride, previousLayer->GetWidth(), previousLayer->GetHeight(), previousLayer->GetDepth());

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetForwardHostMem(true), GetForwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Pool: Forward memory out of range";
	}
#endif

#ifdef _UNITTEST
	//DebugPrint();
#endif
}

void PoolLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for PoolLayer");
}

void PoolLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	std::fill_n(GetBackwardHostMem(false), GetBackwardNodeCount(), (int)0);

	if (cudaMemcpy(backwardDeviceMem, GetBackwardHostMem(false), GetBackwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}

	PoolLayer_Backward(nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, backDataDeviceMem, nodeCount);

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetBackwardHostMem(true), GetBackwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Pool: Backward memory out of range";
	}
#endif

#ifdef _UNITTEST
	//DebugPrint();
#endif
}

double* PoolLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, GetForwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
		}
	}

	return forwardHostMem.get();
}

double* PoolLayer::GetBackwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
}

int* PoolLayer::GetBackDataHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backDataHostMem.get(), backDataDeviceMem, GetBackDataNodeCount() * sizeof(int), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
		}
	}

	return backDataHostMem.get();
}

int PoolLayer::GetBackDataNodeCount()
{
	return nodeCount;
}

double* PoolLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* PoolLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int PoolLayer::GetForwardNodeCount()
{
	return forwardCount;
}

int PoolLayer::GetForwardWidth()
{
	return layerWidth;
}

int PoolLayer::GetForwardHeight()
{
	return layerHeight;
}

int PoolLayer::GetForwardDepth()
{
	return layerDepth;
}

int PoolLayer::GetBackwardNodeCount()
{
	return backwardWidth * backwardHeight * backwardDepth;
}

int PoolLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int PoolLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int PoolLayer::GetBackwardDepth()
{
	return backwardDepth;
}

int PoolLayer::GetWidth()
{
	return layerWidth;
}

int PoolLayer::GetHeight()
{
	return layerHeight;
}

int PoolLayer::GetDepth()
{
	return layerDepth;
}

void PoolLayer::GetLayerData(LayerDataList& layerDataList)
{
	LayerData* layerData = new LayerData[2];

	layerDataList.layerDataCount = 2;
	layerDataList.layerType = LayerType::Pool;
	layerDataList.layerData = layerData;

	layerData->type = LayerDataType::Forward;
	layerData->width = GetForwardWidth();
	layerData->height = GetForwardHeight();
	layerData->depth = GetForwardDepth();
	layerData->data = GetForwardHostMem(true);
	layerData++;

	layerData->type = LayerDataType::Backward;
	layerData->width = backwardWidth;
	layerData->height = backwardHeight;
	layerData->depth = backwardDepth;
	layerData->data = GetBackwardHostMem(true);
	//layerData++;

	//layerData->type = LayerDataType::PoolBackData;
	//layerData->width = nodeCount;
	//layerData->height = 1;
	//layerData->depth = 1;
	//layerData->data = backDataHostMem.get();
}

void PoolLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytesPerSecond)
{
	// TODO: FILL THIS!
}

LayerType PoolLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void PoolLayer::DebugPrint()
{
	std::cout << "pool layer:\r\n";

	std::cout << "forward:\r\n";
	TestUtils::DebugPrintRectangularMemory(GetForwardHostMem(true), GetForwardWidth(), GetForwardHeight(), GetForwardDepth());

	std::cout << "back data:\r\n";
	TestUtils::DebugPrintMemory(GetBackDataHostMem(true), GetBackDataNodeCount());

	std::cout << "backward:\r\n";
	TestUtils::DebugPrintRectangularMemory(GetBackwardHostMem(true), GetBackwardWidth(), GetBackwardHeight(), GetBackwardDepth());
}