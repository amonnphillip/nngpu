#pragma once

struct LayerData
{
public:
	LayerData(char *typeName, double* typeData)
	{
		//strcpy_s((char*)dataType, (rsize_t)20, typeName);
		data = typeData;
	};
	void* Serialize(unsigned int typeNameLength, char *typeName, unsigned int typeDataLength, double* typeData)
	{
		//strcpy_s((char*)dataType, (rsize_t)20, typeName);
		data = typeData;
	}
	unsigned int typeNameLength;
	char* typeName;
	unsigned int typeDataLength;
	double* data;
};
