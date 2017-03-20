#pragma once

extern "C"
{
	__declspec(dllexport) NnGpu* Initialize();
	__declspec(dllexport) void InitializeNetwork(NnGpu* nn);
	__declspec(dllexport) void AddInputLayer(NnGpu* nn, int width, int height, int depth);
	__declspec(dllexport) void AddConvLayer(NnGpu* nn, int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride);
	__declspec(dllexport) void AddReluLayer(NnGpu* nn);
	__declspec(dllexport) void AddPoolLayer(NnGpu* nn, int spatialExtent, int stride);
	__declspec(dllexport) void AddFullyConnected(NnGpu* nn, int size);
	__declspec(dllexport) void AddOutput(NnGpu* nn, int size);
	__declspec(dllexport) void GetLayerType(NnGpu* nn, int layerIndex, int* layerType);
	__declspec(dllexport) void GetLayerCount(NnGpu* nn, int* layerCount);
	__declspec(dllexport) void InitializeTraining(NnGpu* nn);
	__declspec(dllexport) bool TrainNetworkInteration(NnGpu* nn);
	__declspec(dllexport) void DisposeNetwork(NnGpu* nn);
	__declspec(dllexport) void GetLayerDataSize(NnGpu* nn, int layerIndex, int dataType, int* width, int* height, int* depth);
	__declspec(dllexport) void GetLayerData(NnGpu* nn, int layerIndex, int dataType, double* layerData);
}