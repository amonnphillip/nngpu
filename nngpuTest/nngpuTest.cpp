// nngpuTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "nngpu.h"
#include "nngpuwin.h"

int main()
{
	NnGpu* nn = Initialize();
	InitializeNetwork(nn);
	bool testResults = RunUnitTests(nn);


	AddInputLayer(nn, 28, 28, 1);
	AddConvLayer(nn, 3, 3, 1, 32, 1, 1);
	//nn->AddReluLayer(_nn);
	//nn->AddPoolLayer(_nn, 2, 2);
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

	int numIterations = 100;
	while (numIterations > 0)
	{
		TrainNetworkInteration(nn);
		numIterations--;
	}


	DisposeNetwork(nn);
	delete nn;

	delete imageData;
	delete labelData;


    return 0;
}

