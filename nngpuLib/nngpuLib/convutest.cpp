
#include "convutest.h"
#include "convutestreference.h"
#include "testlayer.h"
#include "testutils.h"

bool ConvUTest::Test()
{
	TestLayer* previousLayer = new TestLayer(28, 28, 1);
	previousLayer->ResetForwardAndBackward();

	ConvLayerConfig convConfig = ConvLayerConfig(3, 3, 1, 32, 1, 1);
	ConvLayer* convLayer = new ConvLayer(&convConfig, previousLayer);

	ConvUTestReference* convLayerReference = new ConvUTestReference(&convConfig, previousLayer);

	TestLayer* nextLayer = new TestLayer(28, 28, 1);
	nextLayer->ResetForwardAndBackward();

	convLayer->Forward(previousLayer, nextLayer);

	convLayerReference->ReferenceForward(previousLayer, nextLayer);


	bool isSame = TestUtils::CompareMemory(convLayer->GetForwardHostMem(true), convLayerReference->GetForwardHostMem(true), convLayer->GetForwardNodeCount());

	isSame = TestUtils::CompareMemory(convLayer->GetFilterHostMem(true), convLayerReference->GetFilterHostMem(true), convLayer->GetFilterMemNodeCount());


	convLayer->Backward(previousLayer, nextLayer, 0.01);
	convLayerReference->ReferenceBackward(previousLayer, nextLayer, 0.01);

	isSame = TestUtils::CompareMemory(convLayer->GetBackwardHostMem(true), convLayerReference->GetBackwardHostMem(true), convLayer->GetBackwardNodeCount());

	isSame = TestUtils::CompareMemory(convLayer->GetBackFilterHostMem(true), convLayerReference->GetBackFilterHostMem(true), convLayer->GetBackFilterMemNodeCount());


	return true;
}
