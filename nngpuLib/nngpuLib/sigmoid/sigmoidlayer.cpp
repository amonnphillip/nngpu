#include "sigmoidlayer.h"
#include "layerexception.h"
#include <cassert>
#include "cuda_runtime.h"
#include "layer.h"

extern void SigmoidLayer_Forward(SigmoidNode *node, double *input, double *output, int nodeCount);
extern void SigmoidLayer_Backward(SigmoidNode *node, double* forward, double *input, double *output, int nodeCount, double learnRate);

SigmoidLayer::SigmoidLayer(SigmoidLayerConfig* config, INNetworkLayer* previousLayer)
{
	nodeCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	Layer::Initialize(
		"sigmoid",
		nodeCount,
		nodeCount,
		nodeCount,
		true);


	int forwardNodeCount = previousLayer->GetForwardNodeCount();

	// TODO: MOVE THIS SOME PLACE ELSE!!!!
	SigmoidNode* hnodes = nodeHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		hnodes->weightCount = 2;
		hnodes->weights[0] = 1;
		hnodes->weights[1] = 1;
		hnodes++;
	}

	double* fout = forwardHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*fout = 0;
		fout++;
	}

	double* bout = backwardHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*bout = 0;
		bout++;
	}

	if (cudaMemcpy(nodeDeviceMem, nodeHostMem.get(), nodeCount * sizeof(SigmoidNode), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid cudaMemcpy returned an error");
	}

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid cudaMemcpy returned an error");
	}

}

void SigmoidLayer::Dispose()
{
	Layer::Dispose();
}

void SigmoidLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for Sigmoid layer");
}

void SigmoidLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	SigmoidLayer_Forward(nodeDeviceMem, previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
	}
}

void SigmoidLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for Sigmoid layer");
}

void SigmoidLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	SigmoidLayer_Backward(nodeDeviceMem, forwardDeviceMem, nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}
}

double* SigmoidLayer::GetForwardHostMem()
{
	return forwardHostMem.get();
}

double* SigmoidLayer::GetBackwardHostMem()
{
	return backwardHostMem.get();
}

double* SigmoidLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* SigmoidLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int SigmoidLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int SigmoidLayer::GetBackwardNodeCount()
{
	return nodeCount;
}

std::string SigmoidLayer::GetLayerName()
{
	return Layer::GetLayerName();
}