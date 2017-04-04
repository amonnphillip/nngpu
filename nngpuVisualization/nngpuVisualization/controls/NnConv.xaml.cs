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
    /// Interaction logic for NnConv.xaml
    /// </summary>
    public partial class NnConv : UserControl
    {
        public NnConv()
        {
            InitializeComponent();
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            ImageContainer.Children.Clear();
            FilterImageContainer.Children.Clear();
            BackwardsImageContainer.Children.Clear();

            BitmapSource imageSource = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).ToDepthImage();
            Image image = new Image();
            image.Width = 25 * laterDataGroup.layerData[0].depth;
            image.Height = 25;
            image.Stretch = Stretch.Fill;
            image.Source = imageSource;

            ImageContainer.Children.Add(image);


            BitmapSource backwardsImageSource = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Backward).ToDepthImage();
            Image backwardsImage = new Image();
            backwardsImage.Width = 25 * laterDataGroup.layerData[0].depth;
            backwardsImage.Height = 25;
            backwardsImage.Stretch = Stretch.Fill;
            backwardsImage.Source = backwardsImageSource;

            BackwardsImageContainer.Children.Add(backwardsImage);


            NnGpuLayerData[] filterLayers = laterDataGroup.GetLayersOfType(NnGpuLayerDataType.ConvFilter);
            for (int filterIndex = 0; filterIndex < filterLayers.Length; filterIndex++)
            {
                Image filterImage = new Image();
                filterImage.Width = 25;
                filterImage.Height = 25;
                filterImage.Stretch = Stretch.Fill;
                filterImage.Source = filterLayers[filterIndex].ToImage();
                FilterImageContainer.Children.Add(filterImage);
            }
        }
    }
}
