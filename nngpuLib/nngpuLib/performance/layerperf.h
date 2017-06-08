#pragma once

#include <vector>
#include <memory>

#include <cuda_runtime.h>

#include "layerperfnode.h"

class LayerPerf
{
public:
	LayerPerf();
	~LayerPerf();
	const int MAX_PERF_NODES = 100;
	void Start(unsigned int expectedDataLength);
	void Stop();
	void Clear();
	unsigned int Count();
	void CalculateAverages(unsigned int& aveTime, double& aveBytes);

protected:
	std::unique_ptr<LayerPerfNode> nodeList;
	int nodeIndex = 0;
	bool nodeIndexWrapped = false;
	bool started = false;
	unsigned int bytes;
	cudaEvent_t start;
	cudaEvent_t stop;
};