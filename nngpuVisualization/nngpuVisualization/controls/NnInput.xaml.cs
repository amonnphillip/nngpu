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
    /// Interaction logic for NnInput.xaml
    /// </summary>
    public partial class NnInput : UserControl
    {
        public int LayerIndex { get; set; }

        public NnInput()
        {
            InitializeComponent();
            this.DataContext = this;
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            BitmapSource image = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).ToImage();
            layerInputImg.Source = image;
        }
    }
}
