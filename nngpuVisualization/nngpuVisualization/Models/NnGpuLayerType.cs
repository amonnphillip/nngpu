using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nngpuVisualization
{
    public enum NnGpuLayerType
    {
        Convolution = 0,
        Pool = 1,
        FullyConnected = 2,
        Input = 3,
        Output = 4,
        Relu = 5
    }
}
