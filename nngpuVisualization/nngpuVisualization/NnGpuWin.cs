using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace nngpuVisualization
{
    public class NnGpuWin
    {
        private IntPtr _nn;

        public bool PauseTraining
        {
            get
            {
                return _pauseTraining;
            }
            set
            {
                _pauseTraining = value;
            }
        }
        private bool _pauseTraining = false;

        public bool TrainingComplete
        {
            get
            {
                return _trainingComplete;
            }
        }
        private bool _trainingComplete = false;

        public void InitializeNetwork()
        {
            _nn = NnGpuWin.Initialize();
            NnGpuWin.InitializeNetwork(_nn);
            NnGpuWin.AddInputLayer(_nn, 8, 8, 1);
            NnGpuWin.AddConvLayer(_nn, 3, 3, 1, 4, 1, 1);
            NnGpuWin.AddReluLayer(_nn);
            NnGpuWin.AddPoolLayer(_nn, 2, 2);
            NnGpuWin.AddFullyConnected(_nn, 2);
            NnGpuWin.AddOutput(_nn, 2);

            NnGpuWin.InitializeTraining(_nn);
        }

        public void TrainIteration()
        {
            if (!_pauseTraining)
            {
                _trainingComplete = NnGpuWin.TrainNetworkInteration(_nn);
            }
        }

        public BitmapSource GetNetworkOutputAsImage(int layerIndex, int dataType)
        {
            int width = 0;
            int height = 0;
            int depth = 0;
            NnGpuWin.GetLayerDataSize(_nn, layerIndex, dataType, out width, out height, out depth);

            double[] data = new double[width * height * depth];
            NnGpuWin.GetLayerData(_nn, layerIndex, dataType, data);

            byte[] imageData = new byte[width * height * 4];
            int i = 0;
            for (int x = 0;x < imageData.Length; x+=4)
            {
                byte c = (byte)(data[i] * 255);
                if (c > 255)
                {
                    c = 255;
                }

                imageData[x] = c;
                imageData[x + 1] = c;
                imageData[x + 2] = c;
                imageData[x + 3] = 0xff;

                i++;
            }

            Bitmap bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0,
                                                bitmap.Width,
                                                bitmap.Height),
                                  ImageLockMode.WriteOnly,
                                  bitmap.PixelFormat);

            IntPtr pNative = bmpData.Scan0;
            Marshal.Copy(imageData, 0, pNative, imageData.Length);

            bitmap.UnlockBits(bmpData);

            var handle = bitmap.GetHbitmap();

            return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions()); ;
        }

        public void DisposeNetwork()
        {
            NnGpuWin.DisposeNetwork(_nn);
        }

        public int GetLayerCount()
        {
            int count = 0;
            NnGpuWin.GetLayerCount(_nn, out count);

            return count;
        }

        public int getLayerType(int layerIndex)
        {
            int type = 0;
            NnGpuWin.GetLayerType(_nn, layerIndex, out type);

            return type;
        }

        [DllImport("nngpuLib.dll")]
        private static extern IntPtr Initialize();

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
        public static extern void InitializeTraining(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern bool TrainNetworkInteration(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern void DisposeNetwork(IntPtr nn);

        [DllImport("nngpuLib.dll")]
        public static extern void GetLayerDataSize(IntPtr nn, int layerIndex, int dataType, out int width, out int height, out int depth);

        [DllImport("nngpuLib.dll")]
        public static extern void GetLayerData(IntPtr nn, int layerIndex, int dataType, [In, Out] double[] data);
    }
}
