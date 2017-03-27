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
            _nn = NnGpuWinInterop.Initialize();
            NnGpuWinInterop.InitializeNetwork(_nn);
            NnGpuWinInterop.AddInputLayer(_nn, 8, 8, 1);
            NnGpuWinInterop.AddConvLayer(_nn, 3, 3, 1, 4, 1, 1);
            NnGpuWinInterop.AddReluLayer(_nn);
            NnGpuWinInterop.AddPoolLayer(_nn, 2, 2);
            NnGpuWinInterop.AddFullyConnected(_nn, 2);
            NnGpuWinInterop.AddOutput(_nn, 2);

            NnGpuWinInterop.InitializeTraining(_nn);
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
