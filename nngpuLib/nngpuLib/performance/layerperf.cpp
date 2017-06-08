#include <cassert>

#include "layerperf.h"


LayerPerf::LayerPerf()
{
	nodeList = std::unique_ptr<LayerPerfNode>(new LayerPerfNode[MAX_PERF_NODES]);
}

LayerPerf::~LayerPerf()
{
	if (started)
	{
		Stop();
	}

	nodeList.release();
}

void LayerPerf::Start(unsigned int expectedDataLength)
{
	assert(!started);

	bytes = expectedDataLength;

	started = true;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void LayerPerf::Stop()
{
	assert(started);

	float ms;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	started = false;

	LayerPerfNode* list = nodeList.get();
	list[nodeIndex].Set((unsigned int)ms, bytes);

	nodeIndex++;
	if (nodeIndex >= MAX_PERF_NODES)
	{
		nodeIndex = 0;
		nodeIndexWrapped = true;
	}
}

void LayerPerf::Clear()
{
	assert(started);

	nodeIndex = 0;
	nodeIndexWrapped = false;
}

unsigned int LayerPerf::Count()
{
	if (nodeIndexWrapped)
	{
		return MAX_PERF_NODES;
	} 
	else
	{
		return nodeIndex;
	}
}

void LayerPerf::CalculateAverages(unsigned int& aveTime, double& aveBytes)
{
	LayerPerfNode* list = nodeList.get();
	unsigned int count = Count();
	unsigned int time = 0;
	double bytes = 0;
	for (unsigned int index = 0;index < count;index++)
	{
		unsigned int t = 0;
		double b = 0;
		list->Get(t, b);

		time += t;
		bytes += b;
	}

	time = time / count;
	bytes = bytes / count;

	aveTime = time;
	aveBytes = bytes;
}