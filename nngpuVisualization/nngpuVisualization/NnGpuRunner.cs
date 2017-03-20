using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace nngpuVisualization
{
    class NnGpuRunner
    {
        public delegate void NnGpuRunnerStarted(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerStopped(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerTraningInterationsComplete(NnGpuWin nnGpuWinInstance);

        private static NnGpuWin _nnGpuWin;
        private static bool _workerRunning = false;
        private static BackgroundWorker _worker;

        public static void StartRunner(NnGpuRunnerStarted startedDelegate, NnGpuRunnerStopped stoppedDelegate, NnGpuRunnerTraningInterationsComplete interationCompleteDelegate)
        {
            Debug.Assert(_worker == null);

            if (!_workerRunning)
            {
                _workerRunning = true;

                _nnGpuWin = new NnGpuWin();
                _nnGpuWin.InitializeNetwork();

                _worker = new BackgroundWorker();
                _worker.WorkerReportsProgress = true;

                _worker.DoWork += new DoWorkEventHandler(
                    delegate (object sender, DoWorkEventArgs args)
                    {
                        BackgroundWorker backgroundWorker = sender as BackgroundWorker;

                        startedDelegate(_nnGpuWin);

                        while(!_nnGpuWin.TrainingComplete)
                        {
                            _nnGpuWin.TrainIteration();
                            backgroundWorker.ReportProgress(0);
                            Thread.Sleep(1000);
                        }
                    });

                _worker.ProgressChanged += new ProgressChangedEventHandler(
                    delegate (object sender, ProgressChangedEventArgs args)
                    {
                        interationCompleteDelegate(_nnGpuWin);
                    });

                _worker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(
                    delegate (object sender, RunWorkerCompletedEventArgs args)
                    {
                        stoppedDelegate(_nnGpuWin);
                    });

                _worker.RunWorkerAsync();
            }
        }
    }
}
