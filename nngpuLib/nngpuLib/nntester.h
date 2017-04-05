#pragma once
#include "nnetwork.h"
#include "nntestresult.h"

class NNTester
{
public:
	NNTester();
	~NNTester();
	void Initialize(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength);
	void Iterate(NNetwork* nn, NNTestResult* testresult);
	bool TestingComplete();
	int GetTestingIteration();
	unsigned char* GetImage(int imageIndex);
	unsigned char GetLabel(int labelIndex);

private:
	int iterationCount = 0;

	unsigned char* testingImageData = nullptr;
	unsigned char* testingLabelData = nullptr;
	int testingImageCount;
	int testingImageWidth;
	int testingImageHeight;
	int testingLabelCount;
};
