using nngpuVisualization.CustomMarshal;
using nngpuVisualization.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace nngpuVisualization
{
    class NnGpuWinInterop
    {
        [DllImport("nngpuLib.dll")]
        public static extern IntPtr Initialize();

        [DllImport("nngpuLib.dll")]
        public static extern void InitializeNetwork(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern void AddInputLayer(IntPtr nn, int width, int height, int depth);

        [DllImport("nngpuLib.dll")]
        public static extern void AddConvLayer(IntPtr nn, int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride);

        [DllImport("nngpuLib.dll")]
        public static extern void AddReluLayer(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern void AddPoolLayer(IntPtr nn, int spatialExtent, int stride);

        [DllImport("nngpuLib.dll")]
        public static extern void AddFullyConnected(IntPtr nn, int size);

        [DllImport("nngpuLib.dll")]
        public static extern void AddOutput(IntPtr nn, int size);

        [DllImport("nngpuLib.dll")]
        public static extern void GetLayerType(IntPtr nn, int layerIndex, out int layerType);

        [DllImport("nngpuLib.dll")]
        public static extern void GetLayerCount(IntPtr nn, out int layerCount);

        [DllImport("nngpuLib.dll")]
        public static extern void InitializeTraining(IntPtr nn, [In, MarshalAs(UnmanagedType.LPArray)] byte[] imageData, int imageDataLength, [In, MarshalAs(UnmanagedType.LPArray)] byte[] labelData, int labelDataLength);

        [DllImport("nngpuLib.dll")]
        public static extern bool TrainNetworkInteration(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern void InitializeTesting(IntPtr nn, [In, MarshalAs(UnmanagedType.LPArray)] byte[] imageData, int imageDataLength, [In, MarshalAs(UnmanagedType.LPArray)] byte[] labelData, int labelDataLength);

        [DllImport("nngpuLib.dll")]
        public static extern bool TestNetworkInteration(IntPtr nn, [Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(NNGpuMarshalTestResult))] out NNGpuTestResult result);

        [DllImport("nngpuLib.dll")]
        public static extern void GetTrainingIteration(IntPtr nn, out int interation);

        [DllImport("nngpuLib.dll")]
        public static extern void DisposeNetwork(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern void GetLayerData(IntPtr nn, int layerIndex, [Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(NnGpuMarshalLayerData))] out NnGpuLayerDataGroup data);
    }
}
