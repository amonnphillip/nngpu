#include "poolutestreference.h"
#include "layersize.h"
#include "layerexception.h"
#include <cassert>

extern void PoolLayer_Forward_reference(double *previousLayerForward, double *output, int* backData, int nodeCount, int width, int height, int depth, int stride, int previousLayerWidth, int previousLayerHeight, int previousLayerDepth);
extern void PoolLayer_Backward_reference(double* nextlayerBackward, double *output, int* backwardData, int nodeCount);

PoolUTestReference::PoolUTestReference(PoolLayerConfig* config, INNetworkLayer* previousLayer) :
	PoolLayer(config, previousLayer)
{
}

void PoolUTestReference::ReferenceForward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	double negativeInfinity = -std::numeric_limits<double>::infinity();
	std::fill_n(forwardHostMem.get(), GetForwardNodeCount(), negativeInfinity);
	std::fill_n(GetBackDataHostMem(false), GetBackDataNodeCount(), (int)0);

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), GetForwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backDataDeviceMem, GetBackDataHostMem(false), GetBackDataNodeCount() * sizeof(int), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}

	PoolLayer_Forward_reference(previousLayer->GetForwardDeviceMem(), forwardDeviceMem, backDataDeviceMem, nodeCount, layerWidth, layerHeight, layerDepth, stride, previousLayer->GetWidth(), previousLayer->GetHeight(), previousLayer->GetDepth());
	
	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
	throw std::runtime_error("PoolLayer forward cudaMemcpy returned an error");
	}
	
#ifdef _UNITTEST
	//DebugPrint();
#endif
}

void PoolUTestReference::ReferenceBackward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	std::fill_n(backwardHostMem.get(), GetBackwardNodeCount(), (double)0.0);

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), GetBackwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}

	PoolLayer_Backward_reference(nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, backDataDeviceMem, nodeCount);
	
	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
	throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backDataHostMem.get(), backDataDeviceMem, GetBackDataNodeCount() * sizeof(int), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
	throw std::runtime_error("PoolLayer backward cudaMemcpy returned an error");
	}
	
#ifdef _UNITTEST
	//DebugPrint();
#endif
}