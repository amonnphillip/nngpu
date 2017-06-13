#include <cassert>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#include "softmaxlayer.h"
#include "layerexception.h"
#include "testutils.h"

extern void SoftmaxLayer_Forward(double *previousLayerForward, double *output, double*expDeviceMem, int nodeCount);
extern void SoftmaxLayer_Backward(double* nextlayerBackward, double *output, double* forward, int nodeCount);

SoftmaxLayer::SoftmaxLayer(SoftmaxLayerConfig* config, INNetworkLayer* previousLayer)
{
	layerWidth = previousLayer->GetForwardWidth();
	layerHeight = previousLayer->GetForwardHeight();
	layerDepth = previousLayer->GetForwardDepth();

	backwardWidth = previousLayer->GetForwardWidth();
	backwardHeight = previousLayer->GetForwardHeight();
	backwardDepth = previousLayer->GetForwardDepth();

	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Softmax,
		forwardCount,
		backwardWidth * backwardHeight * backwardDepth,
		nodeCount,
		true);

	expHostMem = std::unique_ptr<double>(new double[nodeCount]);
	std::fill_n(expHostMem.get(), nodeCount, (double)0.0);

	if (cudaMalloc((void**)&expDeviceMem, nodeCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(expDeviceMem, expHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("CudaMemcpy returned an error");
	}
}

void SoftmaxLayer::Dispose()
{
	if (expDeviceMem != nullptr)
	{
		if (cudaFree(expDeviceMem) != cudaError::cudaSuccess)
		{
			throw std::bad_alloc();
		}
		expDeviceMem = nullptr;
	}

	expHostMem.release();

	Layer::Dispose();


}

void SoftmaxLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for PoolLayer");
}

void SoftmaxLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	TestUtils::DebugPrintMemory(previousLayer->GetForwardHostMem(true), previousLayer->GetForwardNodeCount());

	SoftmaxLayer_Forward(previousLayer->GetForwardDeviceMem(), forwardDeviceMem, expDeviceMem, nodeCount);


#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetForwardHostMem(true), GetForwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Softmax: Forward memory out of range";
	}
#endif

#ifdef _UNITTEST
	DebugPrint();
#endif
}

void SoftmaxLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for PoolLayer");
}

void SoftmaxLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	SoftmaxLayer_Backward(nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, forwardDeviceMem, nodeCount);

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetBackwardHostMem(true), GetBackwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Softmax: Backward memory out of range";
	}
#endif

#ifdef _UNITTEST
	DebugPrint();
#endif
}

double* SoftmaxLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, GetForwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Softmax forward cudaMemcpy returned an error");
		}
	}

	return forwardHostMem.get();
}

double* SoftmaxLayer::GetBackwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Softmax backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
}

double* SoftmaxLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* SoftmaxLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int SoftmaxLayer::GetForwardNodeCount()
{
	return forwardCount;
}

int SoftmaxLayer::GetForwardWidth()
{
	return layerWidth;
}

int SoftmaxLayer::GetForwardHeight()
{
	return layerHeight;
}

int SoftmaxLayer::GetForwardDepth()
{
	return layerDepth;
}

int SoftmaxLayer::GetBackwardNodeCount()
{
	return backwardWidth * backwardHeight * backwardDepth;
}

int SoftmaxLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int SoftmaxLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int SoftmaxLayer::GetBackwardDepth()
{
	return backwardDepth;
}

int SoftmaxLayer::GetWidth()
{
	return layerWidth;
}

int SoftmaxLayer::GetHeight()
{
	return layerHeight;
}

int SoftmaxLayer::GetDepth()
{
	return layerDepth;
}

void SoftmaxLayer::GetLayerData(LayerDataList& layerDataList)
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

void SoftmaxLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytesPerSecond)
{
	// TODO: FILL THIS!
}

LayerType SoftmaxLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void SoftmaxLayer::DebugPrint()
{
	std::cout << "Softmax layer:\r\n";

	std::cout << "forward:\r\n";
	TestUtils::DebugPrintMemory(GetForwardHostMem(true), nodeCount);

	std::cout << "backward:\r\n";
	TestUtils::DebugPrintMemory(GetBackwardHostMem(true), nodeCount);
}