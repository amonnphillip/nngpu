#include "stdafx.h"
#include "nngpu.h"
#include "nngpuwin.h"

int main()
{
	// Run unit tests on individual layers
	NnGpu* nn = Initialize();
	InitializeNetwork(nn);
	bool testResults = RunUnitTests(nn);

	if (!testResults)
	{
		DisposeNetwork(nn);
		delete nn;
		return -1;
	}

	// Run tests on the network as a whole
	AddInputLayer(nn, 28, 28, 1);
	AddConvLayer(nn, 3, 3, 32, 1, 1);
	AddReluLayer(nn);
	AddPoolLayer(nn, 2, 2);
	AddConvLayer(nn, 3, 3, 32, 1, 1);
	AddReluLayer(nn);
	AddPoolLayer(nn, 2, 2);
	AddFullyConnected(nn, 10);
	AddOutput(nn, 10);
	
	long imageDataSize = 0;
	unsigned char* imageData = nullptr;
	FILE* imageDataFile = fopen("t10k-images.idx3-ubyte", "r");
	if (imageDataFile != nullptr)
	{
		fseek(imageDataFile, 0, SEEK_END);
		imageDataSize = ftell(imageDataFile);
		fseek(imageDataFile, 0, SEEK_SET);
		imageData = new unsigned char[imageDataSize];
		fread(imageData, 1, imageDataSize, imageDataFile);
		fclose(imageDataFile);
	}

	long labelDataSize = 0;
	unsigned char* labelData = nullptr;
	FILE* labelDataFile = fopen("t10k-labels.idx1-ubyte", "r");
	if (labelDataFile != nullptr)
	{
		fseek(labelDataFile, 0, SEEK_END);
		labelDataSize = ftell(labelDataFile);
		fseek(labelDataFile, 0, SEEK_SET);
		labelData = new unsigned char[labelDataSize];
		fread(labelData, 1, labelDataSize, labelDataFile);
		fclose(labelDataFile);
	}

	InitializeTraining(nn, imageData, imageDataSize, labelData, labelDataSize);

	bool trainError = false;
	int numIterations = 100;
	while (numIterations > 0 && 
		trainError == false)
	{
		try {
			TrainNetworkInteration(nn);
		} 
		catch (...)
		{
			trainError = true;
		}

		numIterations--;
	}

	// Release network resources
	DisposeNetwork(nn);
	delete nn;

	delete imageData;
	delete labelData;

	if (trainError)
	{
		return -1;
	}

    return 0;
}

