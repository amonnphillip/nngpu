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
    /// Interaction logic for NnRelu.xaml
    /// </summary>
    public partial class NnRelu : UserControl
    {
        public NnRelu()
        {
            InitializeComponent();
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            ImageContainer.Children.Clear();
            BackwardImageContainer.Children.Clear();

            NnGpuLayerData forward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward);
            BitmapSource imageSource = forward.ToDepthImage();
            Image image = new Image();
            image.Width = 25 * forward.depth;
            image.Height = 25;
            image.Stretch = Stretch.Fill;
            image.Source = imageSource;

            ImageContainer.Children.Add(image);


            NnGpuLayerData backward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward);
            BitmapSource backwardImageSource = backward.ToDepthImage();
            Image backwardImage = new Image();
            backwardImage.Width = 25 * backward.depth;
            backwardImage.Height = 25;
            backwardImage.Stretch = Stretch.Fill;
            backwardImage.Source = backwardImageSource;

            BackwardImageContainer.Children.Add(backwardImage);
        }
    }
}
