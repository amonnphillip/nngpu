#include "convutestreference.h"
#include "layersize.h"

extern void ConvLayer_ForwardReference(ConvNode *node, double* filters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, double *previousLayerOutput, double *output, int pad);
extern void ConvLayer_BackwardReference(ConvNode *node, double* filters, double* backFilters, LayerSize filterSize, int filterCount, LayerSize layerSize, LayerSize previousLayerSize, LayerSize nextLayerSize, double *previousLayerOutput, double *nextLayerOutput, double *output, int pad, double learnRate);


ConvUTestReference::ConvUTestReference(ConvLayerConfig* config, INNetworkLayer* previousLayer) :
	ConvLayer(config, previousLayer)
{

}

void ConvUTestReference::ReferenceForward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	std::fill_n(forwardHostMem.get(), GetForwardNodeCount(), (double)0.0);

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), GetForwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}

	ConvLayer_ForwardReference(
		nodeDeviceMem,
		filterDeviceMem,
		LayerSize(filterWidth, filterHeight, filterDepth),
		filterCount,
		LayerSize(layerWidth, layerHeight, layerDepth),
		LayerSize(previousLayer->GetForwardWidth(), previousLayer->GetForwardHeight(), previousLayer->GetForwardDepth()),
		previousLayer->GetForwardDeviceMem(),
		forwardDeviceMem,
		pad);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(filterHostMem.get(), filterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer forward cudaMemcpy returned an error");
	}
}

void ConvUTestReference::ReferenceBackward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	std::fill_n(backwardHostMem.get(), GetBackwardNodeCount(), (double)0.0);
	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), GetBackwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}

	std::fill_n(backFilterHostMem.get(), filterSize * filterDepth * filterCount, (double)0.0);
	if (cudaMemcpy(backFilterDeviceMem, backFilterHostMem.get(), filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ConvLayer cudaMemcpy returned an error");
	}

	ConvLayer_BackwardReference(
		nodeDeviceMem,
		filterDeviceMem,
		backFilterDeviceMem,
		LayerSize(filterWidth, filterHeight, filterDepth),
		filterCount,
		LayerSize(layerWidth, layerHeight, layerDepth),
		LayerSize(previousLayer->GetForwardWidth(), previousLayer->GetForwardHeight(), previousLayer->GetForwardDepth()),
		LayerSize(nextLayer->GetBackwardWidth(), nextLayer->GetBackwardHeight(), nextLayer->GetBackwardDepth()),
		previousLayer->GetForwardDeviceMem(),
		nextLayer->GetBackwardDeviceMem(),
		backwardDeviceMem,
		pad,
		learnRate);
	
	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
	throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(nodeHostMem.get(), nodeDeviceMem, nodeCount * sizeof(ConvNode), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
	throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backFilterHostMem.get(), backFilterDeviceMem, filterSize * filterDepth * filterCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
	throw std::runtime_error("ConvLayer backward cudaMemcpy returned an error");
	}
}