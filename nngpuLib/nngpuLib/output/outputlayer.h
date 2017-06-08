#pragma once

#include "outputlayerconfig.h"
#include "innetworklayer.h"
#include "layer.h"
#include "layerdata.h"

struct OutputNode
{
	double output;
};

class OutputLayer : public Layer<OutputNode, double, double, double>, public INNetworkLayer
{
private:
	int backwardWidth = 0;
	int backwardHeight = 0;
	int backwardDepth = 0;
	int nodeCount = 0;

public:
	OutputLayer(OutputLayerConfig* config, INNetworkLayer* previousLayer);
	virtual void Forward(double* input, int inputSize);
	virtual void Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	virtual void Backward(double* input, int inputSize, double learnRate);
	virtual void Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
	virtual void Dispose();
	virtual double* GetForwardHostMem(bool copyFromDevice);
	virtual double* GetBackwardHostMem(bool copyFromDevice);
	virtual double* GetForwardDeviceMem();
	virtual double* GetBackwardDeviceMem();
	virtual int GetForwardNodeCount();
	virtual int GetForwardWidth();
	virtual int GetForwardHeight();
	virtual int GetForwardDepth();
	virtual int GetBackwardNodeCount();
	virtual int GetBackwardWidth();
	virtual int GetBackwardHeight();
	virtual int GetBackwardDepth();
	virtual int GetWidth();
	virtual int GetHeight();
	virtual int GetDepth();
	virtual LayerType GetLayerType();
	virtual void GetLayerData(LayerDataList& layerDataList);
	virtual void GetLayerPerformance(unsigned int& averageTime, double& averageBytes);
	void DebugPrint(double* expected, int expectedCount);
};