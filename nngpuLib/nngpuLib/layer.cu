#include <stdio.h>
#include <stdexcept>

#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

void LayerSynchronize()
{
	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("CUDA syncronize returned an error");
	}
}