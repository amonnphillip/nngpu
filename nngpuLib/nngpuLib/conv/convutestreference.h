#pragma once

#include "convlayer.h"

class ConvUTestReference : public ConvLayer
{
public:
	ConvUTestReference(ConvLayerConfig* config, INNetworkLayer* previousLayer);
	void ReferenceForward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	void ReferenceBackward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
};
