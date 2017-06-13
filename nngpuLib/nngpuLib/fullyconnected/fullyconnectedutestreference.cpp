#include "fullyconnectedutestreference.h"
#include "layersize.h"

extern void FullyConnectedLayer_ForwardReference(FullyConnectedNode *node, double* weights, int weightCount, double *input, double *output, int nodeCount);
extern void FullyConnectedLayer_BackwardReference(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

FullyConnectedUTestReference::FullyConnectedUTestReference(FullyConnectedLayerConfig* config, INNetworkLayer* previousLayer) :
	FullyConnectedLayer(config, previousLayer)
{
}

void FullyConnectedUTestReference::ReferenceForward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	SetMemory(forwardHostMem.get(), forwardDeviceMem, forwardCount, 0);

	FullyConnectedLayer_ForwardReference(nodeDeviceMem, weightsDeviceMem, weightCount, previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("CudaMemcpy returned an error");
	}
}

void FullyConnectedUTestReference::ReferenceBackward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	SetMemory(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount(), 0);

	FullyConnectedLayer_BackwardReference(nodeDeviceMem, weightsDeviceMem, weightCount, forwardDeviceMem, previousLayer->GetForwardDeviceMem(), nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);
}