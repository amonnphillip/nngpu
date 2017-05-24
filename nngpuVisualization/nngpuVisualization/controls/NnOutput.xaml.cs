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
    /// Interaction logic for NnOutput.xaml
    /// </summary>
    public partial class NnOutput : UserControl
    {
        private const double BarUintHeight = 20;
        private const double BarWidth = 40;

        private const int MaxPoints = 100;


        private List<double[]> Outputs { get; set; }

        public NnOutput()
        {
            Outputs = new List<double[]>(100);

            InitializeComponent();
            DataContext = this;
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {
            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            double largest = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).GetLargestDataValue();
            double smallest = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).GetSmallestDataValue();

            double[] layerDataForward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Forward).data;
            double[] layerDataBackward = laterDataGroup.GetLayerOfType(NnGpuLayerDataType.Backward).data;

            Outputs.Insert(0, layerDataBackward);
            if (Outputs.Count > MaxPoints)
            {
                Outputs.RemoveAt(MaxPoints);
            }

            if (double.IsNaN(BarContainer.ActualWidth) ||
                double.IsNaN(BarContainer.ActualHeight))
            {
                return;
            }

            double highAve = 0;
            double lowAve = 0;
            double[] aves = new double[Outputs.Count];
            for (int index = 0; index < Outputs.Count; index++)
            {
                double[] output = Outputs[index];

                double ave = 0;
                for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
                {
                    ave += output[outputIndex];
                }
                ave /= output.Length;
                aves[index] = ave;

                if (ave > highAve)
                {
                    highAve = ave;
                }

                if (ave < lowAve)
                {
                    lowAve = ave;
                }
            }

            double scale = 1;
            if (System.Math.Abs(highAve) > System.Math.Abs(lowAve))
            {
                scale = (BarContainer.Height / System.Math.Abs(highAve)) / 2;
            }
            else
            {
                scale = (BarContainer.Height / System.Math.Abs(lowAve)) / 2;
            }

            if (scale < 0.3)
            {
                scale = 0.3;
            }

            BarContainer.Children.Clear();

            double xscale = this.ActualWidth / ((double)MaxPoints);

            Line baseline = new Line();
            baseline.X1 = 0;
            baseline.Y1 = BarContainer.ActualHeight / 2;
            baseline.X2 = BarContainer.ActualWidth;
            baseline.Y2 = BarContainer.ActualHeight / 2;
            baseline.Stroke = new SolidColorBrush(Color.FromRgb(0, 0, 0));
            baseline.StrokeThickness = 1;
            BarContainer.Children.Add(baseline);

            double lastPointX = 0;
            double lastPointY = 0;
            for (int index = 0; index < Outputs.Count; index++)
            {
                double dataPointX = index * xscale;
                double dataPointY = ((aves[index] * -1) * scale) + (BarContainer.ActualHeight / 2);

                if (index > 0 
                    && !double.IsNaN(lastPointY) 
                    && !double.IsNaN(dataPointY))
                {
                    Line l = new Line();
                    l.X1 = lastPointX;
                    l.Y1 = lastPointY;
                    l.X2 = dataPointX;
                    l.Y2 = dataPointY;
                    l.Stroke = new SolidColorBrush(Color.FromRgb(255, 255, 255));
                    l.StrokeThickness = 1;
                    BarContainer.Children.Add(l);

                    BarContainer.Children.Add(new TextBlock()
                    {
                        Text = Convert.ToString(Math.Round(aves[index], 4)),
                        Margin = new Thickness(dataPointX, dataPointY - 10, 0, 0)
                    });
                }

                lastPointX = dataPointX;
                lastPointY = dataPointY;

            }

            BarContainer.Children.Add(new TextBlock()
            {
                Text = Convert.ToString(Math.Round(highAve, 4)),
                Margin = new Thickness(0, 0, 0, 0)
            });

            BarContainer.Children.Add(new TextBlock()
            {
                Text = Convert.ToString(Math.Round(lowAve, 4)),
                Margin = new Thickness(0, BarContainer.Height - 20, 0, 0)
            });

            BarContainer.InvalidateArrange();
        }
    }
}
