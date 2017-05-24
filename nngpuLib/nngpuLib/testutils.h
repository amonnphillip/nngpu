#pragma once

class TestUtils
{
public:
	static bool CompareMemory(double* src, double* src2, int count)
	{
		for (int i = 0; i < count; i++)
		{
			if (*src != *src2)
			{
				return false;
			}

			src++;
			src2++;
		}

		return true;
	}

	static double SumMemory(double* src, int count)
	{
		double sum = 0;
		for (int i = 0; i < count; i++)
		{
			sum += *src;
			src++;
		}

		return sum;
	}
};