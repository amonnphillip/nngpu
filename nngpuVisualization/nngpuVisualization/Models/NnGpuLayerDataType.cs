using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nngpuVisualization
{
    public enum NnGpuLayerDataType
    {
        Forward = 0,
        Backward = 1,
        ConvForwardFilter = 2,
        ConvBackwardFilter = 3,
        PoolBackData = 4
    }
}
