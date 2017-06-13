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
    /// Interaction logic for NnConv.xaml
    /// </summary>
    public partial class NnConv : UserControl, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        public string BackwardSum
        {
            get
            {
                return _backwardSum;
            }
            set
            {
                _backwardSum = value;
                OnPropertyChanged("BackwardSum");
            }
        }
        private string _backwardSum;

        public string ForwardSum
        {
            get
            {
                return _forwardSum;
            }
            set
            {
                _forwardSum = value;
                OnPropertyChanged("ForwardSum");
            }
        }
        private string _forwardSum;

        public NnConv()
        {
            InitializeComponent();

            DataContext = this;
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            uint averageTimeMs = 0;
            double averageBytes = 0;
            nnGpuWinInstance.GetLayerPerformanceData(layerIndex, out averageTimeMs, out averageBytes);

            Performance timer = new Performance();
            timer.Start();

            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            double largest = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).GetLargestDataValue();
            double smallest = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).GetSmallestDataValue();

            BackwardSum = "Sum: " + laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Backward).Sum();
            ForwardSum = "Sum: " + laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).Sum();

            ImageContainer.Children.Clear();
            FilterImageContainer.Children.Clear();
            BackwardsImageContainer.Children.Clear();

            NnGpuLayerData forward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward);
            BitmapSource imageSource = forward.ToDepthImage();
            Image image = new Image();
            image.Width = 25 * forward.depth;
            image.Height = 25;
            image.Stretch = Stretch.Fill;
            image.Source = imageSource;

            ImageContainer.Children.Add(image);


            NnGpuLayerData backward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Backward);
            BitmapSource backwardsImageSource = backward.ToDepthImage();
            Image backwardsImage = new Image();
            backwardsImage.Width = backward.width;
            backwardsImage.Height = 25;
            backwardsImage.Stretch = Stretch.Fill;
            backwardsImage.Source = backwardsImageSource;

            BackwardsImageContainer.Children.Add(backwardsImage);


            NnGpuLayerData[] filterLayers = laterDataGroup.GetLayersOfType(NnGpuLayerDataType.ConvForwardFilter);
            for (int filterIndex = 0; filterIndex < filterLayers.Length; filterIndex++)
            {
                Image filterImage = new Image();
                filterImage.Width = 25;
                filterImage.Height = 25;
                filterImage.Stretch = Stretch.Fill;
                filterImage.Source = filterLayers[filterIndex].ToImage();
                FilterImageContainer.Children.Add(filterImage);
            }
            long ms = timer.Stop();
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
