#pragma once

#include "vector"
#include "convlayerconfig.h"
#include "innetworklayer.h"
#include "layer.h"
#include "layerdata.h"

class ConvNode
{
public:
	double bias;
};

class ConvLayer : public Layer<ConvNode, double, double, double>, public INNetworkLayer
{
private:
	int nodeCount = 0;
	int forwardCount = 0;
	int backwardCount = 0;
	int layerWidth = 0;
	int layerHeight = 0;
	int layerDepth = 0;
	int pad = 0;
	int stride = 0;
	int filterWidth = 0;
	int filterHeight = 0;
	int filterDepth = 0;
	int filterSize = 0;
	int filterCount = 0;
	std::unique_ptr<double> filterHostMem;
	double* filterDeviceMem;
	std::unique_ptr<double> backFilterHostMem;
	double* backFilterDeviceMem;

public:
	ConvLayer(ConvLayerConfig* config, INNetworkLayer* previousLayer);
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
	virtual int GetWidth();
	virtual int GetHeight();
	virtual int GetDepth();
	virtual LayerType GetLayerType();
	virtual void GetLayerData(LayerDataList& layerDataList);
	double* GetFilterHostMem(bool copyFromDevice);
	void DebugPrint();
};
