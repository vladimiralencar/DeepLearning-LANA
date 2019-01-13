
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
using namespace std;

int main()
{
	int count;
	cudaGetDeviceCount(&count);

	cudaDeviceProp prop;

	for (int i = 0; i < count; ++i)
	{
		cudaGetDeviceProperties(&prop, i);

		cout << "Device " << i << ": " << prop.name << endl;
		cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;

	}

	return 0;
}


