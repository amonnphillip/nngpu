
#include "convutest.h"
#include "convutestreference.h"
#include "testlayer.h"
#include "testutils.h"

bool ConvUTest::Test()
{
	const int LAYER_WIDTH = 28;
	const int LAYER_HEIGHT = 28;
	const int LAYER_DEPTH = 32;
	const int LAYER_FILTERS = 32;

	// Set up out test network
	TestLayer* previousLayer = new TestLayer(LAYER_WIDTH, LAYER_HEIGHT, LAYER_DEPTH);
	previousLayer->ResetForwardAndBackward();

	ConvLayerConfig convConfig = ConvLayerConfig(3, 3, LAYER_FILTERS, 1, 1);
	ConvLayer* convLayer = new ConvLayer(&convConfig, previousLayer);

	ConvUTestReference* convLayerReference = new ConvUTestReference(&convConfig, previousLayer);

	TestLayer* nextLayer = new TestLayer(LAYER_WIDTH, LAYER_HEIGHT, LAYER_DEPTH);
	nextLayer->ResetForwardAndBackward();

	// Test forward
	convLayer->Forward(previousLayer, nextLayer);
	convLayerReference->ReferenceForward(previousLayer, nextLayer);

	int errorx;
	int errory;
	int errord;

	const int NUM_OF_TESTS = 11;
	bool* isSame = new bool[NUM_OF_TESTS];

	isSame[0] = convLayer->GetForwardWidth() == LAYER_WIDTH;
	isSame[1] = convLayer->GetForwardHeight() == LAYER_HEIGHT;
	isSame[2] = convLayer->GetForwardDepth() == LAYER_DEPTH;

	isSame[3] = convLayer->GetBackwardWidth() == LAYER_WIDTH;
	isSame[4] = convLayer->GetBackwardHeight() == LAYER_HEIGHT;
	isSame[5] = convLayer->GetBackwardDepth() == LAYER_DEPTH;

	isSame[6] = TestUtils::CompareRectangularMemory(convLayer->GetForwardHostMem(true), convLayerReference->GetForwardHostMem(true), convLayer->GetForwardWidth(), convLayer->GetForwardHeight(), convLayer->GetForwardDepth(), &errorx, &errory, &errord);
	isSame[7] = TestUtils::CompareMemory(convLayer->GetFilterHostMem(true), convLayerReference->GetFilterHostMem(true), convLayer->GetFilterMemNodeCount());

	// Test backward
	convLayer->Backward(previousLayer, nextLayer, 0.01);
	convLayerReference->ReferenceBackward(previousLayer, nextLayer, 0.01);

	isSame[8] = TestUtils::CompareRectangularMemory(convLayer->GetBackwardHostMem(true), convLayerReference->GetBackwardHostMem(true), convLayer->GetBackwardWidth(), convLayer->GetBackwardHeight(), convLayer->GetBackwardDepth(), &errorx, &errory, &errord);
	isSame[9] = TestUtils::CompareMemory(convLayer->GetBackFilterHostMem(true), convLayerReference->GetBackFilterHostMem(true), convLayer->GetBackFilterMemNodeCount());

	ConvNode* a = convLayer->GetNodeMem(true);
	ConvNode* b = convLayerReference->GetNodeMem(true);
	bool biasSame = true;
	for (int index = 0; index < convLayer->GetForwardNodeCount(); index++)
	{
		if (a->bias != b->bias)
		{
			biasSame = false;
		}
		a++;
		b++;
	}
	isSame[10] = biasSame;

	bool testResult = TestUtils::AllTrue(isSame, NUM_OF_TESTS);
	delete isSame;

	delete previousLayer;
	delete convLayer;
	delete convLayerReference;
	delete nextLayer;

	return testResult;
}
