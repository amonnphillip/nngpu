using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nngpuVisualization
{
    public class NnGpuLayerDataGroup
    {
        public int count;
        public NnGpuLayerType type;
        public NnGpuLayerData[] layerData;

        public NnGpuLayerData GetLayerOfType(NnGpuLayerDataType type)
        {
            NnGpuLayerData matchedLayer = null;

            foreach (NnGpuLayerData dataLayer in layerData)
            {
                if (dataLayer.type == type)
                {
                    matchedLayer = dataLayer;
                    break;
                }
            }

            return matchedLayer;
        }

        public NnGpuLayerData[] GetLayersOfType(NnGpuLayerDataType type)
        {
            List<NnGpuLayerData> matchedLayers = new List<NnGpuLayerData>();

            foreach (NnGpuLayerData dataLayer in layerData)
            {
                if (dataLayer.type == type)
                {
                    matchedLayers.Add(dataLayer);
                }
            }

            return matchedLayers.ToArray();
        }
    }
}
