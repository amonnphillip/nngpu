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
    /// Interaction logic for NnFullyConnected.xaml
    /// </summary>
    public partial class NnFullyConnected : UserControl, INotifyPropertyChanged
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

        public NnFullyConnected()
        {
            InitializeComponent();

            DataContext = this;
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            uint averageTimeMs = 0;
            double averageBytes = 0;
            nnGpuWinInstance.GetLayerPerformanceData(layerIndex, out averageTimeMs, out averageBytes);
            
            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            BackwardSum = "Sum: " + laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Backward).Sum();
            ForwardSum = "Sum: " + laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).Sum();

            double largest = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).GetLargestDataValue();
            double smallest = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).GetSmallestDataValue();

            ImageContainer.Children.Clear();
            BackwardImageContainer.Children.Clear();

            NnGpuLayerData forward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward);
            BitmapSource imageSource = forward.ToImage();
            Image image = new Image();
            //image.Width = 25 * forward.depth;
            image.Height = 25;
            image.Stretch = Stretch.Fill;
            image.Source = imageSource;

            ImageContainer.Children.Add(image);


            NnGpuLayerData backward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Backward);
            BitmapSource backwardImageSource = backward.ToDepthImage();
            Image backwardImage = new Image();
            backwardImage.Width = 25 * backward.depth;
            backwardImage.Height = 25;
            backwardImage.Stretch = Stretch.Fill;
            backwardImage.Source = backwardImageSource;

            BackwardImageContainer.Children.Add(backwardImage);
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
