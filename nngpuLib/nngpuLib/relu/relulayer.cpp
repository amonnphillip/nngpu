#include <cassert>

#include "relulayer.h"
#include "layerexception.h"
#include "testutils.h"

extern void ReluLayer_Forward(double *previousLayerForward, double *output, int nodeCount);
extern void ReluLayer_Backward(double *forward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

ReluLayer::ReluLayer(INNetworkLayer* previousLayer)
{
	backwardWidth = previousLayer->GetForwardWidth();
	backwardHeight = previousLayer->GetForwardHeight();
	backwardDepth = previousLayer->GetForwardDepth();

	layerWidth = previousLayer->GetWidth();
	layerHeight = previousLayer->GetHeight();
	layerDepth = previousLayer->GetDepth();

	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Relu,
		forwardCount,
		backwardWidth * backwardHeight * backwardDepth,
		nodeCount,
		true);
}

void ReluLayer::Dispose()
{
	Layer::Dispose();
}

void ReluLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for ReluLayer layer");
}

void ReluLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	assert(GetForwardNodeCount() == previousLayer->GetForwardNodeCount());

	std::fill_n(GetForwardHostMem(false), GetForwardNodeCount(), (double)0.0);

	if (cudaMemcpy(forwardDeviceMem, GetForwardHostMem(false), GetForwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer forward cudaMemcpy returned an error");
	}

	ReluLayer_Forward(previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);
/*
	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer forward cudaMemcpy returned an error");
	}
*/

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetForwardHostMem(true), GetForwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Relu: Forward memory out of range";
	}
#endif
}

void ReluLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for ReluLayer layer");
}

void ReluLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	assert(GetBackwardNodeCount() == nextLayer->GetBackwardNodeCount());

	std::fill_n(GetBackwardHostMem(false), GetBackwardNodeCount(), (double)0.0);

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), GetBackwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer backward cudaMemcpy returned an error");
	}

	ReluLayer_Backward(forwardDeviceMem, nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);
/*
	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer backward cudaMemcpy returned an error");
	}
*/

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetBackwardHostMem(true), GetBackwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Relu: Forward memory out of range";
	}
#endif
}

double* ReluLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, GetForwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("ReluLayer forward cudaMemcpy returned an error");
		}
	}

	return forwardHostMem.get();
}

double* ReluLayer::GetBackwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("ReluLayer backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
}

double* ReluLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* ReluLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int ReluLayer::GetForwardNodeCount()
{
	return forwardCount;
}

int ReluLayer::GetForwardWidth()
{
	return layerWidth;
}

int ReluLayer::GetForwardHeight()
{
	return layerHeight;
}

int ReluLayer::GetForwardDepth()
{
	return layerDepth;
}

int ReluLayer::GetBackwardNodeCount()
{
	return backwardWidth * backwardHeight * backwardDepth;
}

int ReluLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int ReluLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int ReluLayer::GetBackwardDepth()
{
	return backwardDepth;
}

int ReluLayer::GetWidth()
{
	return layerWidth;
}

int ReluLayer::GetHeight()
{
	return layerHeight;
}

int ReluLayer::GetDepth()
{
	return layerDepth;
}

void ReluLayer::GetLayerData(LayerDataList& layerDataList)
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
	layerData->width = GetForwardWidth();
	layerData->height = GetForwardHeight();
	layerData->depth = GetForwardDepth();
	layerData->data = GetBackwardHostMem(true);
}

void ReluLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytesPerSecond)
{
	// TODO: FILL THIS!
}

LayerType ReluLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void ReluLayer::DebugPrint()
{
	std::cout << "Relu layer:\r\n";

	std::cout << "forward:\r\n";
	TestUtils::DebugPrintRectangularMemory(GetForwardHostMem(true), GetForwardWidth(), GetForwardHeight(), GetForwardDepth());

	std::cout << "backward:\r\n";
	TestUtils::DebugPrintRectangularMemory(GetBackwardHostMem(true), GetBackwardWidth(), GetBackwardHeight(), GetBackwardDepth());
}