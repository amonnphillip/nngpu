#pragma once
#include <memory>
#include <cassert>
#include <stdexcept>
#include "layertype.h"
#include "cuda_runtime.h"

extern void LayerSynchronize();

template<typename nodetype, typename inputtype, typename forwardtype, typename backwardtype>
class Layer
{
protected:
	LayerType layerType;
	std::unique_ptr<nodetype> nodeHostMem;
	std::unique_ptr<forwardtype> forwardHostMem;
	std::unique_ptr<backwardtype> backwardHostMem;
	nodetype* nodeDeviceMem = nullptr;
	forwardtype* forwardDeviceMem = nullptr;
	backwardtype* backwardDeviceMem = nullptr;

	void Initialize(LayerType layerType, int forwardSize, int backwardSize, int nodeSize, bool hasDeviceMemory)
	{
		assert(nodeHostMem.get() == nullptr);
		assert(forwardHostMem.get() == nullptr);
		assert(backwardHostMem.get() == nullptr);
		assert(nodeDeviceMem == nullptr);
		assert(forwardDeviceMem == nullptr);
		assert(backwardDeviceMem == nullptr);

		this->layerType = layerType;

		nodeHostMem = std::unique_ptr<nodetype>(new nodetype[nodeSize]); // TODO: MAKE THIS OPTIONAL?
		if (nodeHostMem.get() == nullptr)
		{
			throw std::bad_alloc();
		}

		forwardHostMem = std::unique_ptr<forwardtype>(new forwardtype[forwardSize]);
		if (forwardHostMem.get() == nullptr)
		{
			throw std::bad_alloc();
		}

		backwardHostMem = std::unique_ptr<backwardtype>(new backwardtype[backwardSize]);
		if (backwardHostMem.get() == nullptr)
		{
			throw std::bad_alloc();
		}

		if (hasDeviceMemory)
		{
			if (nodeSize != 0)
			{
				if (cudaMalloc((void**)&nodeDeviceMem, nodeSize * sizeof(nodetype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}

			if (forwardSize != 0)
			{
				if (cudaMalloc((void**)&forwardDeviceMem, forwardSize * sizeof(forwardtype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}

			if (backwardSize != 0)
			{
				if (cudaMalloc((void**)&backwardDeviceMem, backwardSize * sizeof(backwardtype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}
		}
	}

	virtual void Dispose()
	{
		if (nodeDeviceMem != nullptr)
		{
			if (cudaFree(nodeDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}

		if (forwardDeviceMem != nullptr)
		{
			if (cudaFree(forwardDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}

		if (backwardDeviceMem != nullptr)
		{
			if (cudaFree(backwardDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}
	}

public:
	virtual LayerType GetLayerType()
	{
		return layerType;
	}

	inline void Synchronize()
	{
		LayerSynchronize();
	}
	
	virtual void SetMemory(double* hostMem, double* deviceMem, int count, double value)
	{
		std::fill_n(hostMem, count, (double)value);

		if (cudaMemcpy(deviceMem, hostMem, count * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
		{
			throw std::runtime_error("CudaMemcpy returned an error");
		}
	}
};