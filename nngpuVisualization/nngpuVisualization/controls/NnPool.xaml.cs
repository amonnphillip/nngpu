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
    /// Interaction logic for NnPool.xaml
    /// </summary>
    public partial class NnPool : UserControl
    {
        public NnPool()
        {
            InitializeComponent();
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            ImageContainer.Children.Clear();
            BackwardImageContainer.Children.Clear();

            BitmapSource imageSource = laterDataGroup.layerData[0].ToDepthImage();
            Image image = new Image();
            image.Width = 25 * laterDataGroup.layerData[0].depth;
            image.Height = 25;
            image.Stretch = Stretch.Fill;
            image.Source = imageSource;

            ImageContainer.Children.Add(image);

            BitmapSource backwardImageSource = laterDataGroup.layerData[1].ToDepthImage();
            Image backwardImage = new Image();
            backwardImage.Width = 25 * laterDataGroup.layerData[1].depth;
            backwardImage.Height = 25;
            backwardImage.Stretch = Stretch.Fill;
            backwardImage.Source = backwardImageSource;

            BackwardImageContainer.Children.Add(backwardImage);
        }
    }
}
