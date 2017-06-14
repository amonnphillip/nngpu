#include <cassert>
#include <iostream>
#include "fullyconnectedlayer.h"
#include "layerexception.h"
#ifdef _DEBUG
#include "testutils.h"
#endif

extern void FullyConnectedLayer_Forward(FullyConnectedNode *node, double* weights, int weightCount, double *input, double *output, int nodeCount);
extern void FullyConnectedLayer_Backward(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayerConfig* config, INNetworkLayer* previousLayer)
{
	layerWidth = config->GetWidth();
	layerHeight = config->GetHeight();
	layerDepth = config->GetDepth();

	backwardWidth = previousLayer->GetForwardWidth();
	backwardHeight = previousLayer->GetForwardHeight();
	backwardDepth = previousLayer->GetForwardDepth();

	weightCount = previousLayer->GetForwardNodeCount();
	forwardCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	nodeCount = forwardCount;
	Layer::Initialize(
		LayerType::FullyConnected,
		forwardCount,
		backwardWidth * backwardHeight * backwardDepth,
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

	std::fill_n(weightsHostMem.get(), weightCount * nodeCount, (double)1 / (double)weightCount);

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
	SetMemory(forwardHostMem.get(), forwardDeviceMem, forwardCount, 0);

	layerPerf.Start(layerWidth * layerHeight * layerDepth);
	FullyConnectedLayer_Forward(nodeDeviceMem, weightsDeviceMem, weightCount, previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);
	layerPerf.Stop();

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Forward cudaMemcpy returned an error");
	}	

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetForwardHostMem(true), GetForwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "Fullyconnected: Forward memory out of range";
	}
#endif

#ifdef _UNITTEST
	//DebugPrint();
#endif
}

void FullyConnectedLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for Sigmoid layer");
}

void FullyConnectedLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	assert(previousLayer->GetForwardNodeCount() == weightCount);
	assert(nextLayer->GetBackwardNodeCount() == nodeCount);

	SetMemory(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount(), 0);

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(previousLayer->GetForwardHostMem(true), previousLayer->GetForwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "FullyConnected: Backward memory out of range";
	}
#endif

	//layerPerf.Start(layerWidth * layerHeight * layerDepth);
	FullyConnectedLayer_Backward(nodeDeviceMem, weightsDeviceMem, weightCount, forwardDeviceMem, previousLayer->GetForwardDeviceMem(), nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);
	//layerPerf.Stop();

#ifdef _UNITTEST
	if (TestUtils::HasElementOutOfRange(GetBackwardHostMem(true), GetBackwardNodeCount(), -100, 100))
	{
		DebugPrint();
		throw "FullyConnected: Backward memory out of range";
	}

	if (TestUtils::HasElementOutOfRange(weightsHostMem.get(), weightCount * nodeCount, -100, 100))
	{
		DebugPrint();
		throw "FullyConnected: Backward memory out of range";
	}
#endif

#ifdef _UNITTEST
	//DebugPrint();
#endif
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
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
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
	return backwardWidth * backwardHeight * backwardDepth;
}

int FullyConnectedLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int FullyConnectedLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int FullyConnectedLayer::GetBackwardDepth()
{
	return backwardDepth;
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

FullyConnectedNode* FullyConnectedLayer::GetNodeMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(nodeHostMem.get(), nodeDeviceMem, nodeCount * sizeof(FullyConnectedNode), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("CudaMemcpy returned an error");
		}
	}

	return nodeHostMem.get();
}

double* FullyConnectedLayer::GetWeightHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(weightsHostMem.get(), weightsDeviceMem, weightCount * nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("CudaMemcpy returned an error");
		}
	}

	return weightsHostMem.get();
}

void FullyConnectedLayer::GetLayerData(LayerDataList& layerDataList)
{
	LayerData* layerData = new LayerData[2];

	layerDataList.layerDataCount = 2;
	layerDataList.layerType = LayerType::Input;
	layerDataList.layerData = layerData;

	layerData->type = LayerDataType::Forward;
	layerData->width = GetForwardWidth();
	layerData->height = GetForwardHeight();
	layerData->depth = GetForwardDepth();
	layerData->data = GetForwardHostMem(true);
	layerData++;

	layerData->type = LayerDataType::Backward;
	layerData->width = GetBackwardWidth();
	layerData->height = GetBackwardHeight();
	layerData->depth = GetBackwardDepth();
	layerData->data = GetBackwardHostMem(true);
}

void FullyConnectedLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytes)
{
	layerPerf.CalculateAverages(averageTime, averageBytes);
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
	FullyConnectedNode* node = GetNodeMem(true);
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
