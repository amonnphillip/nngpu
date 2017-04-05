using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace nngpuVisualization.CustomMarshal
{
    class NnGpuMarshalLayerData : ICustomMarshaler
    {
        public IntPtr MarshalManagedToNative(object managedObj)
        {
            return IntPtr.Zero;
        }

        public object MarshalNativeToManaged(IntPtr obj)
        {
            NnGpuLayerDataGroup layerDataGroup = new NnGpuLayerDataGroup();
            
            layerDataGroup.count = Marshal.ReadInt32(obj);
            obj += 4;
            layerDataGroup.type = (NnGpuLayerType)Marshal.ReadInt32(obj);
            layerDataGroup.layerData = new NnGpuLayerData[layerDataGroup.count];

            obj += 4;
            for (int index = 0;index < layerDataGroup.count; index ++)
            {
                NnGpuLayerData layerData = new NnGpuLayerData();
                layerData.type = (NnGpuLayerDataType)Marshal.ReadInt32(obj);
                obj += 4;
                layerData.width = Marshal.ReadInt32(obj);
                obj += 4;
                layerData.height = Marshal.ReadInt32(obj);
                obj += 4;
                layerData.depth = Marshal.ReadInt32(obj);
                obj += 4;
                double[] data = new double[layerData.width * layerData.height * layerData.depth];
                Marshal.Copy(obj, data, 0, layerData.width * layerData.height * layerData.depth);
                layerData.data = data;

                layerDataGroup.layerData[index] = layerData;

                obj += layerData.width * layerData.height * layerData.depth * 8;
            }

            return layerDataGroup;
        }

        public void CleanUpNativeData(IntPtr pNativeData)
        {
            Marshal.FreeHGlobal(pNativeData);
        }

        public void CleanUpManagedData(object managedObj)
        {
        }

        public int GetNativeDataSize()
        {
            return -1;
        }

        public static ICustomMarshaler GetInstance(string cookie)
        {
            return new NnGpuMarshalLayerData();
        }
    }
}
