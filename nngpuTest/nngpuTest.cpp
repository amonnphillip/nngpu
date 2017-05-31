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
    return 0;
}

