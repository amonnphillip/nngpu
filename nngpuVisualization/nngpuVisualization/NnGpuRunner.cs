using nngpuVisualization.Models;
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
        public delegate void NnGpuRunnerTrainingStarted(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerTraningInterationsComplete(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerTrainingStopped(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerTestingStarted(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerTestingInterationsComplete(NnGpuWin nnGpuWinInstance);
        public delegate void NnGpuRunnerTestingStopped(NnGpuWin nnGpuWinInstance);

        public enum RunnerStatus
        {
            None,
            Training,
            Testing
        };

        public static RunnerStatus Status
        {
            get
            {
                return _status;
            }
            set
            {
                _status = value;
            }
        }
        private static RunnerStatus _status = RunnerStatus.None;

        private static NnGpuWin _nnGpuWin;
        private static bool _workerRunning = false;
        private static BackgroundWorker _worker;

        public static int UpdateInterval
        {
            get
            {
                return _updateInterval;
            }
            set
            {
                _updateInterval = value;
            }
        }
#if DEBUG
        private static int _updateInterval = 10;
#else
        private static int _updateInterval = 100;
#endif

        private static int _currentInterval = 0;

        private static int _threadDelayMs = 10;

        private static bool _shownInitialProgress = false;


        public static void StartRunner(
            NnGpuRunnerTrainingStarted trainingStartedDelegate, 
            NnGpuRunnerTrainingStopped trainingStoppedDelegate, 
            NnGpuRunnerTraningInterationsComplete trainingInterationCompleteDelegate,
            NnGpuRunnerTestingStarted testingStartedDelegate,
            NnGpuRunnerTestingStopped testingStoppedDelegate,
            NnGpuRunnerTestingInterationsComplete testingInterationCompleteDelegate)
        {
            Debug.Assert(_worker == null);

            if (!_workerRunning)
            {
                _workerRunning = true;

                _nnGpuWin = new NnGpuWin();

#if false
                bool testResults = _nnGpuWin.RunUnitTests();
                if (!testResults)
                {
                    throw new Exception("NN Unit test results failed!");
                }
#endif
                _nnGpuWin.InitializeNetwork();

                _worker = new BackgroundWorker();
                _worker.WorkerReportsProgress = true;

                _worker.DoWork += new DoWorkEventHandler(
                    delegate (object sender, DoWorkEventArgs args)
                    {
                        Status = RunnerStatus.Training;

                        BackgroundWorker backgroundWorker = sender as BackgroundWorker;

                        trainingStartedDelegate(_nnGpuWin);

                        while(!_nnGpuWin.TrainingComplete)
                        {
                            Performance timer = new Performance();
                            timer.Start();
                            _nnGpuWin.TrainIteration();
                            long ms = timer.Stop(); // TODO: MAKE THIS VALUE ACCESSIBLE


                            if (!_shownInitialProgress)
                            {
                                backgroundWorker.ReportProgress(0);
                                _shownInitialProgress = true;
                            }

                            _currentInterval++;
                            if (_currentInterval >= _updateInterval)
                            {
                                _currentInterval = 0;
                                backgroundWorker.ReportProgress(0);
                            }

                            if (_threadDelayMs > 0)
                            {
                                Thread.Sleep(_threadDelayMs);
                            }
                        }

                        trainingStoppedDelegate(_nnGpuWin);


                        Status = RunnerStatus.Testing;

                        _nnGpuWin.InitializeTesting();

                        testingStartedDelegate(_nnGpuWin);

                        while (!_nnGpuWin.TestingComplete)
                        {

                            _nnGpuWin.TestIteration();


                            _currentInterval++;
                            if (_currentInterval >= _updateInterval)
                            {
                                _currentInterval = 0;
                                backgroundWorker.ReportProgress(0);
                            }

                            if (_threadDelayMs > 0)
                            {
                                Thread.Sleep(_threadDelayMs);
                            }
                        }

                        testingStoppedDelegate(_nnGpuWin);

                    });

                _worker.ProgressChanged += new ProgressChangedEventHandler(
                    delegate (object sender, ProgressChangedEventArgs args)
                    {
                        if (_status == RunnerStatus.Training)
                        {
                            trainingInterationCompleteDelegate(_nnGpuWin);
                        } else if (_status == RunnerStatus.Testing)
                        {
                            testingInterationCompleteDelegate(_nnGpuWin);
                        }
                    });

                _worker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(
                    delegate (object sender, RunWorkerCompletedEventArgs args)
                    {

                    });

                _worker.RunWorkerAsync();
            }
        }
    }
}
