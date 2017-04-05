#include "relulayer.h"
#include "layerexception.h"
#include <cassert>

extern void ReluLayer_Forward(ReluNode *node, double *previousLayerForward, double *output, int nodeCount);
extern void ReluLayer_Backward(ReluNode *node, double *forward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

ReluLayer::ReluLayer(INNetworkLayer* previousLayer)
{
	backwardCount = previousLayer->GetForwardNodeCount();

	layerWidth = previousLayer->GetWidth();
	layerHeight = previousLayer->GetHeight();
	layerDepth = previousLayer->GetDepth();

	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Relu,
		forwardCount,
		backwardCount,
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
	double* forward = forwardHostMem.get();
	for (int index = 0; index < forwardCount; index++)
	{
		*forward = 0;
		forward++;
	}

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), forwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer forward cudaMemcpy returned an error");
	}

	ReluLayer_Forward(nodeDeviceMem, previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);
/*
	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer forward cudaMemcpy returned an error");
	}
*/
}

void ReluLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for ReluLayer layer");
}

void ReluLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	double* backward = backwardHostMem.get();
	for (int index = 0; index < backwardCount; index++)
	{
		*backward = 0;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), backwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer backward cudaMemcpy returned an error");
	}

	ReluLayer_Backward(nodeDeviceMem, forwardDeviceMem, nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);
/*
	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer backward cudaMemcpy returned an error");
	}
*/
}

double* ReluLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
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
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
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
	return backwardCount;
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

LayerType ReluLayer::GetLayerType()
{
	return Layer::GetLayerType();
}