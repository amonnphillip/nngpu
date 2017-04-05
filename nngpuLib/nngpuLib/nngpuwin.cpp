#include <windows.h>
#include "nngpu.h"
#include "nngpuwin.h"
#include "layerdata.h"

int MarshalSize(int width, int height, int depth)
{
	return (sizeof(double) * width * height * depth) + (sizeof(int) * 4);
}

int ToMarshalFormat(void* dest, int type, int width, int height, int depth, double* data)
{
	int* intPtr = (int*)dest;
	*intPtr = type;
	intPtr++;
	*intPtr = width;
	intPtr++;
	*intPtr = height;
	intPtr++;
	*intPtr = depth;
	intPtr++;
	double* doublePtr = (double*)intPtr;
	memcpy(doublePtr, data, sizeof(double) * width * height * depth);

	return MarshalSize(width, height, depth);
}

NnGpu* Initialize()
{
	return new NnGpu();
}

void InitializeNetwork(NnGpu* nn)
{
	nn->InitializeNetwork();
}

void AddInputLayer(NnGpu* nn, int width, int height, int depth)
{
	nn->AddInputLayer(width, height, depth);
}

void AddConvLayer(NnGpu* nn, int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride)
{
	nn->AddConvLayer(filterWidth, filterHeight, filterDepth, filterCount, pad, stride);
}

void AddReluLayer(NnGpu* nn)
{
	nn->AddReluLayer();
}

void AddPoolLayer(NnGpu* nn, int spatialExtent, int stride)
{
	nn->AddPoolLayer(spatialExtent, stride);
}

void AddFullyConnected(NnGpu* nn, int size)
{
	nn->AddFullyConnected(size);
}

void AddOutput(NnGpu* nn, int size)
{
	nn->AddOutput(size);
}

void GetLayerType(NnGpu* nn, int layerIndex, int* layerType)
{
	nn->GetLayerType(layerIndex, layerType);
}

void GetLayerCount(NnGpu* nn, int* layerCount)
{
	nn->GetLayerCount(layerCount);
}

void InitializeTraining(NnGpu* nn, unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength)
{
	nn->InitializeTraining(imageData, imageDataLength, labelData, labelDataLength);
}

bool TrainNetworkInteration(NnGpu* nn)
{
	return nn->TrainNetworkInteration();
}

void InitializeTesting(NnGpu* nn, unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength)
{
	nn->InitializeTesting(imageData, imageDataLength, labelData, labelDataLength);
}

bool TestNetworkInteration(NnGpu* nn, NNTestResult** testresult)
{
	*testresult = (NNTestResult*)GlobalAlloc(GMEM_FIXED, sizeof(NNTestResult));

	return nn->TestNetworkInteration(*testresult);
}

void GetTrainingIteration(NnGpu* nn, int* interation)
{
	*interation = nn->GetTrainingIteration();
}

void DisposeNetwork(NnGpu* nn)
{
	nn->DisposeNetwork();
}

void GetLayerData(NnGpu* nn, int layerIndex, int** data)
{
	LayerDataList layerDataList;
	nn->GetLayerData(layerIndex, LayerDataType::Forward, layerDataList);

	int size = 8;
	for (int index = 0; index < layerDataList.layerDataCount; index++)
	{
		LayerData* ld = &layerDataList.layerData[index];
		size += MarshalSize(ld->width, ld->height, ld->depth);
	}

	int* mem = (int*)GlobalAlloc(GMEM_FIXED, size);
	*data = (int*)mem;

	*mem = layerDataList.layerDataCount;
	mem++;
	*mem = layerDataList.layerType;
	mem++;

	for (int index = 0; index < layerDataList.layerDataCount; index++)
	{
		LayerData* ld = &layerDataList.layerData[index];
		mem += (ToMarshalFormat((void*)mem, ld->type, ld->width, ld->height, ld->depth, ld->data) / sizeof(int));
	}

	layerDataList.CleanUp();
}





