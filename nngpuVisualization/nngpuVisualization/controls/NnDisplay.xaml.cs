using System;
using System.Collections.Generic;
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
    public partial class NnDisplay : UserControl
    {
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

        private List<UserControl> _layerControls;

        public NnDisplay()
        {
            _layerControls = new List<UserControl>();

            InitializeComponent();

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

                });
        }
    }
}
