#include <cassert>
#include <iostream>
#include "inputlayerconfig.h"
#include "inputlayer.h"
#include "layerexception.h"
#include "cuda_runtime.h"
#include "layer.h"

InputLayer::InputLayer(InputLayerConfig* config, INNetworkLayer* previousLayer)
{
	width = config->GetWidth();
	height = config->GetHeight();
	depth = config->GetDepth();
	nodeCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	Layer::Initialize(
		LayerType::Input,
		nodeCount,
		0,
		0,
		true);
}

void InputLayer::Dispose()
{
	Layer::Dispose();
}

void InputLayer::Forward(double* input, int inputSize)
{
	assert(inputSize == nodeCount);

	// TODO: Maybe we dont need to copy here?
	memcpy(forwardHostMem.get(), input, nodeCount * sizeof(double));

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("InputLayer forward cudaMemcpy returned an error");
	}
}

void InputLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	throw LayerException("Forward variant not valid for InputLayer layer");
}

void InputLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for InputLayer layer");
}

void InputLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	throw LayerException("Backward variant not valid for InputLayer layer");
}

double* InputLayer::GetForwardHostMem(bool copyFromDevice)
{
	return forwardHostMem.get();
}

double* InputLayer::GetBackwardHostMem(bool copyFromDevice)
{
	return nullptr;
}

double* InputLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* InputLayer::GetBackwardDeviceMem()
{
	return nullptr;
}

int InputLayer::GetForwardWidth()
{
	return width;
}

int InputLayer::GetForwardHeight()
{
	return height;
}

int InputLayer::GetForwardDepth()
{
	return depth;
}

int InputLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int InputLayer::GetBackwardNodeCount()
{
	return 0;
}

int InputLayer::GetWidth()
{
	return width;
}

int InputLayer::GetHeight()
{
	return height;
}

int InputLayer::GetDepth()
{
	return depth;
}

LayerType InputLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void InputLayer::GetLayerData(LayerDataList& layerDataList)
{
	LayerData* layerData = new LayerData[1];

	layerDataList.layerDataCount = 1;
	layerDataList.layerType = LayerType::Input;
	layerDataList.layerData = layerData;

	layerData->type = LayerDataType::Forward;
	layerData->width = GetForwardWidth();
	layerData->height = GetForwardHeight();
	layerData->depth = GetForwardDepth();
	layerData->data = GetForwardHostMem(true);
}

void InputLayer::DebugPrint()
{
	double* forward = GetForwardHostMem(false);
	int forwardCount = GetForwardNodeCount();

	std::cout << "input:\r\n";
	for (int index = 0; index < forwardCount; index++)
	{
		std::cout << forward[index] << " ";
	}
}