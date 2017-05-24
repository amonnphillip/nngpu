#include "poollayer.h"
#include "layerexception.h"
#include <cassert>

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
	double* forward = forwardHostMem.get();
	double negativeInfinity = -std::numeric_limits<double>::infinity();
	for (int index = 0; index < forwardCount; index++)
	{
		*forward = negativeInfinity;
		forward++;
	}

	int* backData = backDataHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*backData = 0;
		backData++;
	}

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), forwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backDataDeviceMem, backDataHostMem.get(), nodeCount * sizeof(int), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	PoolLayer_Forward(previousLayer->GetForwardDeviceMem(), forwardDeviceMem, backDataDeviceMem, nodeCount, layerWidth, layerHeight, layerDepth, stride, previousLayer->GetWidth(), previousLayer->GetHeight(), previousLayer->GetDepth());
/*
	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}
*/
}

void PoolLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for PoolLayer");
}

void PoolLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	double* backward = backwardHostMem.get();
	int backwardCount = GetBackwardNodeCount();
	for (int index = 0; index < backwardCount; index++)
	{
		*backward = 0;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), backwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}

	PoolLayer_Backward(nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, backDataDeviceMem, nodeCount);
/*
	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backDataHostMem.get(), backDataDeviceMem, nodeCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}
*/
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
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
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

LayerType PoolLayer::GetLayerType()
{
	return Layer::GetLayerType();
}