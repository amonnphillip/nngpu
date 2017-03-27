#include "poollayer.h"
#include "layerexception.h"
#include <cassert>

extern void PoolLayer_Forward(double *previousLayerForward, double *output, int* backData, int nodeCount, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth);
extern void PoolLayer_Backward(double* nextlayerBackward, double *output, int* backwardData, int nodeCount);

PoolLayer::PoolLayer(PoolLayerConfig* config, INNetworkLayer* previousLayer)
{
	layerWidth = (previousLayer->GetWidth() - config->GetSpatialExtent()) / config->GetStride() + 1;
	layerHeight = (previousLayer->GetHeight() - config->GetSpatialExtent()) / config->GetStride() + 1;
	layerDepth = previousLayer->GetDepth();
	spatiallExtent = config->GetSpatialExtent();
	stride = config->GetStride();

	backwardCount = previousLayer->GetForwardNodeCount();
	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Pool,
		forwardCount,
		backwardCount,
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
		*backData = 1;
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
	double* forward = forwardHostMem.get();
	double negativeInfinity = -std::numeric_limits<double>::infinity();
	for (int index = 0; index < forwardCount; index++)
	{
		*forward = negativeInfinity;
	}

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), forwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	PoolLayer_Forward(previousLayer->GetForwardDeviceMem(), forwardDeviceMem, backDataDeviceMem, nodeCount, layerWidth, layerHeight, layerDepth, stride, previousLayer->GetWidth(), previousLayer->GetHeight(), previousLayer->GetDepth());

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}
}

void PoolLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for PoolLayer");
}

void PoolLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
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

	PoolLayer_Backward(nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, backDataDeviceMem, nodeCount);

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}
}

double* PoolLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
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
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
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
	return backwardCount;
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

LayerType PoolLayer::GetLayerType()
{
	return Layer::GetLayerType();
}