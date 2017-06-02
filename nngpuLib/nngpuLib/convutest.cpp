
#include "convutest.h"
#include "convutestreference.h"
#include "testlayer.h"
#include "testutils.h"

bool ConvUTest::Test()
{
	// Set up out test network
	TestLayer* previousLayer = new TestLayer(28, 28, 1);
	previousLayer->ResetForwardAndBackward();

	ConvLayerConfig convConfig = ConvLayerConfig(3, 3, 1, 32, 1, 1);
	ConvLayer* convLayer = new ConvLayer(&convConfig, previousLayer);

	ConvUTestReference* convLayerReference = new ConvUTestReference(&convConfig, previousLayer);

	TestLayer* nextLayer = new TestLayer(28, 28, 1);
	nextLayer->ResetForwardAndBackward();

	// Test forward
	convLayer->Forward(previousLayer, nextLayer);
	convLayerReference->ReferenceForward(previousLayer, nextLayer);

	int errorx;
	int errory;
	int errord;

	const int NUM_OF_TESTS = 4;
	bool* isSame = new bool[NUM_OF_TESTS];
	isSame[0] = TestUtils::CompareRectangularMemory(convLayer->GetForwardHostMem(true), convLayerReference->GetForwardHostMem(true), convLayer->GetForwardWidth(), convLayer->GetForwardHeight(), convLayer->GetForwardDepth(), &errorx, &errory, &errord);
	isSame[1] = TestUtils::CompareMemory(convLayer->GetFilterHostMem(true), convLayerReference->GetFilterHostMem(true), convLayer->GetFilterMemNodeCount());

	// Test backward
	convLayer->Backward(previousLayer, nextLayer, 0.01);
	convLayerReference->ReferenceBackward(previousLayer, nextLayer, 0.01);

	isSame[2] = TestUtils::CompareRectangularMemory(convLayer->GetBackwardHostMem(true), convLayerReference->GetBackwardHostMem(true), convLayer->GetBackwardWidth(), convLayer->GetBackwardHeight(), convLayer->GetBackwardDepth(), &errorx, &errory, &errord);
	isSame[3] = TestUtils::CompareMemory(convLayer->GetBackFilterHostMem(true), convLayerReference->GetBackFilterHostMem(true), convLayer->GetBackFilterMemNodeCount());

	bool testResult = TestUtils::AllTrue(isSame, NUM_OF_TESTS);
	delete isSame;

	delete previousLayer;
	delete convLayer;
	delete convLayerReference;
	delete nextLayer;

	return testResult;
}
