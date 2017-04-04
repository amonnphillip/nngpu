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
using System.IO.Compression;
using System.IO;

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

        public byte[] LoadAndDecompressFile(string filePathAndName)
        {
            byte[] decompressedData;
            FileInfo zippedImages = new FileInfo(filePathAndName);
            using (FileStream fileStream = zippedImages.OpenRead())
            {
                using (GZipStream decompressedStream = new GZipStream(fileStream, CompressionMode.Decompress))
                {
                    using (MemoryStream decompressedMem = new MemoryStream())
                    {
                        decompressedStream.CopyTo(decompressedMem);
                        decompressedData = decompressedMem.ToArray();
                    }
                }
            }

            return decompressedData;
        }

        public void InitializeNetwork()
        {


            byte[] imageData = LoadAndDecompressFile("data\\train-images-idx3-ubyte.gz");
            byte[] labelData = LoadAndDecompressFile("data\\train-labels-idx1-ubyte.gz");





            _nn = NnGpuWinInterop.Initialize();
            NnGpuWinInterop.InitializeNetwork(_nn);
            NnGpuWinInterop.AddInputLayer(_nn, 28, 28, 1);
            NnGpuWinInterop.AddConvLayer(_nn, 3, 3, 1, 32, 1, 1);
            NnGpuWinInterop.AddReluLayer(_nn);
            NnGpuWinInterop.AddPoolLayer(_nn, 2, 2);
            NnGpuWinInterop.AddFullyConnected(_nn, 10);
            NnGpuWinInterop.AddOutput(_nn, 10);

            NnGpuWinInterop.InitializeTraining(_nn, imageData, imageData.Length, labelData, labelData.Length);
        }

        public void TrainIteration()
        {
            if (!_pauseTraining)
            {
                _trainingComplete = NnGpuWinInterop.TrainNetworkInteration(_nn);
            }
        }

        public void DisposeNetwork()
        {
            NnGpuWinInterop.DisposeNetwork(_nn);
        }

        public int GetLayerCount()
        {
            int count = 0;
            NnGpuWinInterop.GetLayerCount(_nn, out count);

            return count;
        }

        public int getLayerType(int layerIndex)
        {
            int type = 0;
            NnGpuWinInterop.GetLayerType(_nn, layerIndex, out type);

            return type;
        }

        public NnGpuLayerDataGroup GetLayerData(int layerIndex)
        {
            NnGpuLayerDataGroup layerDataGroup;

            NnGpuWinInterop.GetLayerData(_nn, layerIndex, out layerDataGroup);

            return layerDataGroup;
        }

        public int GetTrainingIteration()
        {
            int interation = 0;
            NnGpuWinInterop.GetTrainingIteration(_nn, out interation);

            return interation;
        }
    }
}
