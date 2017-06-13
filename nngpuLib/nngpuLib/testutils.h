#pragma once
#include <iostream>

class TestUtils
{
public:
	static bool AreSame(double a, double b) {
		return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
	}

	static bool CompareMemory(double* src, double* src2, int count)
	{
		for (int i = 0; i < count; i++)
		{
			if (!AreSame(*src, *src2))
			{
				return false;
			}

			src++;
			src2++;
		}

		return true;
	}

	static bool CompareMemory(int* src, int* src2, int count)
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

	static bool CompareRectangularMemory(double* src, double* src2, int sizeX, int sizeY, int sizeD, int* errorSizeX, int* errorSizeY, int* errorSizeD)
	{
		for (int d = 0; d < sizeD; d++)
		{
			for (int y = 0; y < sizeY; y++)
			{
				for (int x = 0; x < sizeX; x++)
				{
					if (!AreSame(*src, *src2))
					{
						*errorSizeX = x;
						*errorSizeY = y;
						*errorSizeD = d;
						return false;
					}

					src++;
					src2++;
				}
			}
		}

		return true;
	}

	template<typename T>
	static T SumMemory(T* src, int count)
	{
		T sum = 0;
		for (int i = 0; i < count; i++)
		{
			sum += *src;
			src++;
		}

		return sum;
	}

	static bool AllTrue(bool* src, int count)
	{
		bool allTrue = true;
		for (int i = 0; i < count; i++)
		{
			if (*src == false)
			{
				allTrue = false;
				break;
			}
			src++;
		}

		return allTrue;
	}

	static void GradualFill(double* src, int count)
	{
		double v = -1;
		double inc = 2 / (double)count;
		for (int i = 0; i < count; i++) {
			*src = v;
			src++;
			v += inc;
		}
	}

	static bool HasElementOutOfRange(double* src, int count, double lowValue, double highValue)
	{
		for (int i = 0; i < count; i++) {
			if (*src < lowValue || 
				*src > highValue)
			{
				return true;
			}
			src++;
		}

		return false;
	}

	template<typename T>
	static void DebugPrintRectangularMemory(T* src, int width, int height, int depth)
	{
		for (int d = 0;d < depth; d++)
		{
			for (int y = 0;y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					std::cout << *src << " "; 
					src++;
				}
				std::cout << "\r\n";
			}
			std::cout << "\r\n";
		}

		std::cout.flush();
	}

	template<typename T>
	static void DebugPrintMemory(T* src, int count)
	{
		for (int i = 0;i < count;i++)
		{
			std::cout << *src << " ";
			src++;
		}
		std::cout << "\r\n";
		std::cout.flush();
	}
};