using nngpuVisualization.Models;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace nngpuVisualization.controls
{
    /// <summary>
    /// Interaction logic for NnDisplay.xaml
    /// </summary>
    public partial class NnDisplay : UserControl, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        private enum LayerType
        {
            Convolution = 0,
            Pool,
            FullyConnected,
            Input,
            Output,
            Relu
        };

        public NnGpuWin NnNetwork { get; set; }

        public string IterationText
        {
            get
            {
                return _iterationText;
            }
            set
            {
                _iterationText = value;
                OnPropertyChanged("IterationText");
            }
        }
        private string _iterationText;

        public string TestIterationText
        {
            get
            {
                return _testIterationText;
            }
            set
            {
                _testIterationText = value;
                OnPropertyChanged("TestIterationText");
            }
        }
        private string _testIterationText;

        public string TestAccuracyText
        {
            get
            {
                return _testAccuracyText;
            }
            set
            {
                _testAccuracyText = value;
                OnPropertyChanged("TestAccuracyText");
            }
        }
        private string _testAccuracyText;

        private List<UserControl> _layerControls;

        public NnDisplay()
        {
            _layerControls = new List<UserControl>();

            IterationText = "ddd";

            InitializeComponent();

            DataContext = this;

            NnGpuRunner.StartRunner(
                delegate (NnGpuWin nnGpuWinInstance)
                {
                    // Started callback
                    int layerCount = nnGpuWinInstance.GetLayerCount();

                    for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                    {
                        int layerType = nnGpuWinInstance.getLayerType(layerIndex);

                        switch (layerType)
                        {
                            case (int)LayerType.Input:
                                Dispatcher.Invoke(() => {
                                    NnInput control = new NnInput();
                                    LayerContainer.Children.Add(control);
                                    _layerControls.Add(control);
                                });
                                break;
                            case (int)LayerType.Convolution:
                                Dispatcher.Invoke(() => {
                                    NnConv control = new NnConv();
                                    LayerContainer.Children.Add(control);
                                    _layerControls.Add(control);
                                });
                                break;
                            case (int)LayerType.FullyConnected:
                                Dispatcher.Invoke(() => {
                                    NnFullyConnected control = new NnFullyConnected();
                                    LayerContainer.Children.Add(control);
                                    _layerControls.Add(control);
                                });
                                break;
                            case (int)LayerType.Relu:
                                Dispatcher.Invoke(() => {
                                    NnRelu control = new NnRelu();
                                    LayerContainer.Children.Add(control);
                                    _layerControls.Add(control);
                                });
                                break;
                            case (int)LayerType.Pool:
                                Dispatcher.Invoke(() => {
                                    NnPool control = new NnPool();
                                    LayerContainer.Children.Add(control);
                                    _layerControls.Add(control);
                                });
                                break;
                            case (int)LayerType.Output:
                                Dispatcher.Invoke(() => {
                                    NnOutput control = new NnOutput();
                                    LayerContainer.Children.Add(control);
                                    _layerControls.Add(control);
                                });
                                break;
                            default:
                                Dispatcher.Invoke(() => {
                                    LayerContainer.Children.Add(new Rectangle());
                                    _layerControls.Add(null);
                                });

                                break;
                        }
                        
                    }
                },
                delegate (NnGpuWin nnGpuWinInstance)
                {
                    // End callback
                },
                delegate (NnGpuWin nnGpuWinInstance)
                {
                    Dispatcher.Invoke(() =>
                    {
                        this.IterationText = "Training iteraiton: " + nnGpuWinInstance.GetTrainingIteration();
                        this.UpdateLayout();
                    });

                    // Traning iteration
                    int layerCount = nnGpuWinInstance.GetLayerCount();

                    for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                    {
                        int layerType = nnGpuWinInstance.getLayerType(layerIndex);

                        switch (layerType)
                        {
                            case (int)LayerType.Input:
                                Dispatcher.Invoke(() => {
                                    NnInput control = _layerControls[layerIndex] as NnInput;
                                    control.Update(nnGpuWinInstance, layerIndex);
                                });
                                break;
                            case (int)LayerType.Convolution:
                                Dispatcher.Invoke(() => {
                                    NnConv control = _layerControls[layerIndex] as NnConv;
                                    control.Update(nnGpuWinInstance, layerIndex);
                                });
                                break;
                            case (int)LayerType.FullyConnected:
                                Dispatcher.Invoke(() => {
                                    NnFullyConnected control = _layerControls[layerIndex] as NnFullyConnected;
                                    control.Update(nnGpuWinInstance, layerIndex);
                                });
                                break;
                            case (int)LayerType.Relu:
                                Dispatcher.Invoke(() => {
                                    NnRelu control = _layerControls[layerIndex] as NnRelu;
                                    control.Update(nnGpuWinInstance, layerIndex);
                                });
                                break;
                            case (int)LayerType.Pool:
                                Dispatcher.Invoke(() => {
                                    NnPool control = _layerControls[layerIndex] as NnPool;
                                    control.Update(nnGpuWinInstance, layerIndex);
                                });
                                break;
                            case (int)LayerType.Output:
                                Dispatcher.Invoke(() => {
                                    NnOutput control = _layerControls[layerIndex] as NnOutput;
                                    control.Update(nnGpuWinInstance, layerIndex);
                                });
                                break;
                        }
                    }

                    Dispatcher.Invoke(() => {
                        this.InvalidateVisual();
                    });

                }, 
                delegate (NnGpuWin nnGpuWinInstance)
                {
                    // Training start
                },
                delegate (NnGpuWin nnGpuWinInstance)
                {
                    // Training stop

                },
                delegate (NnGpuWin nnGpuWinInstance)
                {
                    // Training iterate
                    TestIterationText = "Tests: " + nnGpuWinInstance.TestsPerformed;
                    TestAccuracyText = "Test Accuracy: " + nnGpuWinInstance.CorrectTestPredictions;
                }
                );
        }

        protected void OnPropertyChanged(string name)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(name));
            }
        }
    }
}
