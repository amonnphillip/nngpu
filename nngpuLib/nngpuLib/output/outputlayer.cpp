#include <cassert>
#include <iostream>
#include "outputlayer.h"
#include "layerexception.h"
#include "cuda_runtime.h"
#include "layer.h"


OutputLayer::OutputLayer(OutputLayerConfig* config, INNetworkLayer* previousLayer)
{
	nodeCount = config->GetWidth() * config->GetHeight() * config->GetDepth();

	// TODO: LOOK AT THE BACK VALUES
	backwardWidth = previousLayer->GetForwardWidth();
	backwardHeight = previousLayer->GetForwardHeight();
	backwardDepth = previousLayer->GetForwardDepth();

	Layer::Initialize(
		LayerType::Output,
		nodeCount,
		nodeCount,
		0,
		true);
}

void OutputLayer::Dispose()
{
	Layer::Dispose();
}

void OutputLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for OutputLayer layer");
}

void OutputLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	assert(previousLayer->GetForwardNodeCount() == nodeCount);

	memcpy(forwardHostMem.get(), previousLayer->GetForwardHostMem(true), nodeCount * sizeof(double));

	double* forward = forwardHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		if (isnan(*forward) ||
			isinf(*forward))
		{
			*forward = 0;
		}
		forward++;
	}
}

void OutputLayer::Backward(double* input, int inputSize, double learnRate)
{
	assert(inputSize == nodeCount);

	double* backward = backwardHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*backward = *input;
		if (isnan(*backward) ||
			isinf(*backward))
		{
			*backward = 0;
		}
		input++;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("OutputLayer backward cudaMemcpy returned an error");
	}

#ifdef _UNITTEST
	//DebugPrint(nullptr, 0);
#endif
}

void OutputLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	throw LayerException("Backward variant not valid for OutputLayer layer");
}

double* OutputLayer::GetForwardHostMem(bool copyFromDevice)
{
	return forwardHostMem.get();
}

double* OutputLayer::GetBackwardHostMem(bool copyFromDevice)
{
	return backwardHostMem.get();
}

double* OutputLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* OutputLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int OutputLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int OutputLayer::GetForwardWidth()
{
	return nodeCount;
}

int OutputLayer::GetForwardHeight()
{
	return 1;
}

int OutputLayer::GetForwardDepth()
{
	return 1;
}

int OutputLayer::GetBackwardNodeCount()
{
	return backwardWidth * backwardHeight * backwardDepth;
}

int OutputLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int OutputLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int OutputLayer::GetBackwardDepth()
{
	return backwardDepth;
}

int OutputLayer::GetWidth()
{
	return nodeCount;
}

int OutputLayer::GetHeight()
{
	return 1;
}

int OutputLayer::GetDepth()
{
	return 1;
}

LayerType OutputLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void OutputLayer::GetLayerData(LayerDataList& layerDataList)
{
	LayerData* layerData = new LayerData[2];

	layerDataList.layerDataCount = 2;
	layerDataList.layerType = LayerType::Output;
	layerDataList.layerData = layerData;

	layerData[0].type = LayerDataType::Forward;
	layerData[0].width = GetForwardWidth();
	layerData[0].height = GetForwardHeight();
	layerData[0].depth = GetForwardDepth();
	layerData[0].data = GetForwardHostMem(true);

	layerData[1].type = LayerDataType::Backward;
	layerData[1].width = GetBackwardNodeCount();
	layerData[1].height = 1;
	layerData[1].depth = 1;
	layerData[1].data = backwardHostMem.get();
}

void OutputLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytesPerSecond)
{
	// TODO: FILL THIS!
}

void OutputLayer::DebugPrint(double* expected, int expectedCount)
{
	assert(expectedCount == GetForwardNodeCount() || (expected == nullptr && expectedCount == 0));

	double* forward = GetForwardHostMem(false);
	int forwardCount = GetForwardNodeCount();
	std::cout << "output:\r\n";
	for (int index = 0; index < forwardCount; index++)
	{
		std::cout << forward[index] << " ";
	}

	std::cout << "\r\n";
	double* backward = GetBackwardHostMem(false);
	int backwardCount = GetBackwardNodeCount();
	std::cout << "backward:\r\n";
	for (int index = 0; index < backwardCount; index++)
	{
		std::cout << backward[index] << " ";
	}

	std::cout << "\r\n";
	if (expected != nullptr)
	{
		std::cout << "\r\n";
		std::cout << "expected:\r\n";
		for (int index = 0; index < forwardCount; index++)
		{
			std::cout << expected[index] << " ";
		}
	}
}