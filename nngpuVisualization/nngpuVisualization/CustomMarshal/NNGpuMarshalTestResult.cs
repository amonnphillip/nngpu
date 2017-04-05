using nngpuVisualization.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace nngpuVisualization.CustomMarshal
{
    class NNGpuMarshalTestResult : ICustomMarshaler
    {
        public IntPtr MarshalManagedToNative(object managedObj)
        {
            return IntPtr.Zero;
        }

        public object MarshalNativeToManaged(IntPtr obj)
        {
            NNGpuTestResult testResult = new NNGpuTestResult();

            testResult.expected = Marshal.ReadInt32(obj);
            obj += 4;
            testResult.predicted = Marshal.ReadInt32(obj);

            return testResult;
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
            return new NNGpuMarshalTestResult();
        }
    }
}
