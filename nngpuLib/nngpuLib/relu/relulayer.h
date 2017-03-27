#pragma once

#include "innetworklayer.h"
#include "relulayerconfig.h"
#include "layer.h"
#include "layerdata.h"

class ReluNode
{
};

class ReluLayer : public Layer<ReluNode, double, double, double>, public INNetworkLayer
{
private:
	int nodeCount = 0;
	int forwardCount = 0;
	int backwardCount = 0;
	int layerWidth = 0;
	int layerHeight = 0;
	int layerDepth = 0;

public:
	ReluLayer(INNetworkLayer* previousLayer);
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
	virtual void GetLayerData(LayerDataList& layerDataList);
	virtual LayerType GetLayerType();
};

