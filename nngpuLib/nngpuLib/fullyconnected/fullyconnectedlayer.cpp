#include <cassert>
#include <iostream>
#include "fullyconnectedlayer.h"
#include "layerexception.h"

extern void FullyConnectedLayer_Forward(FullyConnectedNode *node, double* weights, int weightCount, double *input, double *output, int nodeCount);
extern void FullyConnectedLayer_Backward(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayerConfig* config, INNetworkLayer* previousLayer)
{
	layerWidth = config->GetWidth();
	layerHeight = config->GetHeight();
	layerDepth = config->GetDepth();

	backwardCount = previousLayer->GetForwardNodeCount();
	weightCount = previousLayer->GetForwardNodeCount();
	forwardCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	nodeCount = forwardCount;
	Layer::Initialize(
		LayerType::FullyConnected,
		forwardCount,
		backwardCount,
		nodeCount,
		true);

	FullyConnectedNode* nodes = nodeHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		nodes->bias = 0;
		nodes++;
	}

	if (cudaMemcpy(nodeDeviceMem, nodeHostMem.get(), nodeCount * sizeof(FullyConnectedNode), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer cudaMemcpy returned an error");
	}

	weightsHostMem = std::unique_ptr<double>(new double[weightCount * nodeCount]);
	if (weightsHostMem.get() == nullptr)
	{
		throw std::bad_alloc();
	}

	double* weights = weightsHostMem.get();
	for (int index = 0; index < weightCount * nodeCount; index++)
	{
		*weights = (double)1 / (double)weightCount;
		weights++;
	}

	if (cudaMalloc((void**)&weightsDeviceMem, weightCount * nodeCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(weightsDeviceMem, weightsHostMem.get(), weightCount * nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer cudaMemcpy returned an error");
	}
}


void FullyConnectedLayer::Dispose()
{
	if (weightsDeviceMem != nullptr)
	{
		if (cudaFree(weightsDeviceMem) != cudaError::cudaSuccess)
		{
			throw std::bad_alloc();
		}
	}

	Layer::Dispose();
}

void FullyConnectedLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for Sigmoid layer");
}

void FullyConnectedLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	FullyConnectedLayer_Forward(nodeDeviceMem, weightsDeviceMem, weightCount, previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
	}
}

void FullyConnectedLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for Sigmoid layer");
}

void FullyConnectedLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	double* backward = backwardHostMem.get();
	for (int index = 0; index < backwardCount; index++)
	{
		*backward = 0;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), backwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}

	FullyConnectedLayer_Backward(nodeDeviceMem, weightsDeviceMem, weightCount, forwardDeviceMem, previousLayer->GetForwardDeviceMem(), nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);

	if (cudaMemcpy(weightsHostMem.get(), weightsDeviceMem, weightCount * nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(nodeHostMem.get(), nodeDeviceMem, nodeCount * sizeof(FullyConnectedNode), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}
}

double* FullyConnectedLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
		}
	}

	return forwardHostMem.get();
}

double* FullyConnectedLayer::GetBackwardHostMem(bool copyFromDevice = false)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
}

double* FullyConnectedLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* FullyConnectedLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int FullyConnectedLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int FullyConnectedLayer::GetForwardWidth()
{
	return layerWidth;
}

int FullyConnectedLayer::GetForwardHeight()
{
	return layerHeight;
}

int FullyConnectedLayer::GetForwardDepth()
{
	return layerDepth;
}

int FullyConnectedLayer::GetBackwardNodeCount()
{
	return backwardCount;
}

double* FullyConnectedLayer::GetWeightsForNode(int index)
{
	assert(index >= 0);
	assert(index < nodeCount);

	return weightsHostMem.get() + (weightCount * index);
}

int FullyConnectedLayer::GetWeightCount()
{
	return weightCount;
}

int FullyConnectedLayer::GetWidth()
{
	return nodeCount;
}

int FullyConnectedLayer::GetHeight()
{
	return 1;
}

int FullyConnectedLayer::GetDepth()
{
	return 1;
}

LayerType FullyConnectedLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

FullyConnectedNode* FullyConnectedLayer::GetNodeMem()
{
	return nodeHostMem.get();
}

void FullyConnectedLayer::GetLayerData(LayerDataList& layerDataList)
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

void FullyConnectedLayer::DebugPrint()
{
	int nodeCount = GetForwardNodeCount();
	int weightCount = GetWeightCount();

	std::cout << "fully connected layer:\r\n";

	std::cout << "weights:\r\n";
	for (int index = 0; index < nodeCount; index++)
	{
		double* weight = GetWeightsForNode(index);

		for (int weightIndex = 0; weightIndex < weightCount; weightIndex++)
		{
			if (weightIndex + 1 != weightCount)
			{
				std::cout << *weight << " ";
			}
			else
			{
				std::cout << *weight << " : ";
			}
			weight++;
		}
	}

	std::cout << "\r\n";
	std::cout << "bias:\r\n";
	FullyConnectedNode* node = GetNodeMem();
	for (int index = 0; index < nodeCount; index++)
	{
		if (index + 1 != nodeCount)
		{
			std::cout << node->bias << " ";
		}
		else
		{
			std::cout << node->bias << " ";
		}
		node++;
	}

	std::cout << "\r\n";

	std::cout << "forward:\r\n";
	double* output = GetForwardHostMem(false);
	for (int index = 0; index < nodeCount; index++)
	{
		std::cout << *output << " ";
		output++;
	}

	std::cout << "\r\n\r\n";
}
