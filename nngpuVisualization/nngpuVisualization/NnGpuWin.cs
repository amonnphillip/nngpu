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
using nngpuVisualization.Models;

namespace nngpuVisualization
{
    public class NnGpuWin
    {
        private IntPtr _nn;

        public bool Pause
        {
            get
            {
                return _pause;
            }
            set
            {
                _pause = value;
            }
        }
        private bool _pause = false;

        public bool TrainingComplete
        {
            get
            {
                return _trainingComplete;
            }
        }
        private bool _trainingComplete = false;

        public bool TestingComplete
        {
            get
            {
                return _testingComplete;
            }
        }
        private bool _testingComplete = false;

        private List<NNGpuTestResult> _testResults;

        public int CorrectTestPredictions
        {
            get
            {
                return _correctPredictions;
            }
        }
        private int _correctPredictions;

        public int TestsPerformed
        {
            get
            {
                return _testResults.Count;
            }
        }

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
            NnGpuWinInterop.AddConvLayer(_nn, 3, 3, 32, 1, 1);
            NnGpuWinInterop.AddReluLayer(_nn);
            NnGpuWinInterop.AddPoolLayer(_nn, 2, 2);
            NnGpuWinInterop.AddConvLayer(_nn, 3, 3, 32, 1, 1);
            NnGpuWinInterop.AddReluLayer(_nn);
            NnGpuWinInterop.AddPoolLayer(_nn, 2, 2);
            NnGpuWinInterop.AddFullyConnected(_nn, 10);
            NnGpuWinInterop.AddSoftmax(_nn, 10);
            NnGpuWinInterop.AddOutput(_nn, 10);

            NnGpuWinInterop.InitializeTraining(_nn, imageData, imageData.Length, labelData, labelData.Length);
        }

        public void InitializeTesting()
        {
            _testResults = new List<NNGpuTestResult>();
            _correctPredictions = 0;

            byte[] imageData = LoadAndDecompressFile("data\\t10k-images-idx3-ubyte.gz");
            byte[] labelData = LoadAndDecompressFile("data\\t10k-labels-idx1-ubyte.gz");

            NnGpuWinInterop.InitializeTesting(_nn, imageData, imageData.Length, labelData, labelData.Length);
        }

        public void TrainIteration()
        {
            if (!_pause)
            {
                _trainingComplete = NnGpuWinInterop.TrainNetworkInteration(_nn);
            }
        }

        public NNGpuTestResult TestIteration()
        {
            if (!_pause)
            {
                NNGpuTestResult result;
                _testingComplete = NnGpuWinInterop.TestNetworkInteration(_nn, out result);

                _testResults.Add(result);

                if (result.expected == result.predicted)
                {
                    _correctPredictions++;
                }

                return result;
            }

            return null;
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

        public void GetLayerPerformanceData(int layerIndex, out uint averageTimeMs, out double averageBytes)
        {
            NnGpuWinInterop.GetLayerPerformanceData(_nn, layerIndex, out averageTimeMs, out averageBytes);
        }

        public bool RunUnitTests()
        {
            return NnGpuWinInterop.RunUnitTests(_nn);
        }
    }
}
