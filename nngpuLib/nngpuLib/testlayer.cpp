#include <algorithm>
#include <iostream>
#include "testlayer.h"
#include "layersize.h"
#include "layerexception.h"
#include <cassert>
#include "cuda_runtime.h"
#include "layer.h"
#include "testutils.h"

TestLayer::TestLayer(int width, int height, int depth)
{
	layerWidth = width;
	layerHeight = height;
	layerDepth = depth;

	backwardWidth = width;
	backwardHeight = height;
	backwardDepth = depth;

	forwardCount = layerWidth * layerHeight * layerDepth;
	nodeCount = forwardCount;

	Layer::Initialize(
		LayerType::Test,
		forwardCount,
		backwardWidth * backwardHeight * backwardDepth,
		nodeCount,
		true);
}

void TestLayer::Dispose()
{
	Layer::Dispose();
}

void TestLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for ConvLayer layer");
}

void TestLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	throw LayerException("Not implemented for TestLayer");
}

void TestLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Not implemented for TestLayer");
}

void TestLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	throw LayerException("Not implemented for TestLayer");
}

double* TestLayer::GetForwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("TestLayer forward cudaMemcpy returned an error");
		}
	}

	return forwardHostMem.get();
}

double* TestLayer::GetBackwardHostMem(bool copyFromDevice)
{
	if (copyFromDevice)
	{
		if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, GetBackwardNodeCount() * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("TestLayer backward cudaMemcpy returned an error");
		}
	}

	return backwardHostMem.get();
}

double* TestLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* TestLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int TestLayer::GetForwardNodeCount()
{
	return forwardCount;
}

int TestLayer::GetForwardWidth()
{
	return layerWidth;
}

int TestLayer::GetForwardHeight()
{
	return layerHeight;
}

int TestLayer::GetForwardDepth()
{
	return layerDepth;
}

int TestLayer::GetBackwardNodeCount()
{
	return backwardWidth * backwardHeight * backwardDepth;
}

int TestLayer::GetBackwardWidth()
{
	return backwardWidth;
}

int TestLayer::GetBackwardHeight()
{
	return backwardHeight;
}

int TestLayer::GetBackwardDepth()
{
	return backwardDepth;
}

int TestLayer::GetWidth()
{
	return layerWidth;
}

int TestLayer::GetHeight()
{
	return layerHeight;
}

int TestLayer::GetDepth()
{
	return layerDepth;
}

LayerType TestLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void TestLayer::GetLayerData(LayerDataList& layerDataList)
{
	throw LayerException("Not implemented for TestLayer");
}

void TestLayer::GetLayerPerformance(unsigned int& averageTime, double& averageBytes)
{

}

void TestLayer::ResetForwardAndBackward()
{
	//std::fill_n(forwardHostMem.get(), GetForwardNodeCount(), (double)1.0);
	TestUtils::GradualFill(forwardHostMem.get(), GetForwardNodeCount());
	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), GetForwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("TestLayer forward cudaMemcpy returned an error");
	}

	//std::fill_n(backwardHostMem.get(), GetBackwardNodeCount(), (double)1.0);
	TestUtils::GradualFill(backwardHostMem.get(), GetBackwardNodeCount());
	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), GetBackwardNodeCount() * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("TestLayer backward cudaMemcpy returned an error");
	}
}
