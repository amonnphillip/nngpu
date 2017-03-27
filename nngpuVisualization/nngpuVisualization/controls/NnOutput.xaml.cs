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

        public NnOutput()
        {
            InitializeComponent();
            this.DataContext = this;
        }

        public void Update(NnGpuWin nnGpuWinInstance, int layerIndex)
        {

            NnGpuLayerDataGroup laterDataGroup = nnGpuWinInstance.GetLayerData(layerIndex);

            double[] layerData = laterDataGroup.layerData[0].data;

            BarContainer.Children.Clear();

            for(int index = 0;index < layerData.Length;index ++)
            {
                double value = layerData[index];

                double barValue = value;
                double marginh = 0;
                if (barValue >= 0)
                {
                    if (barValue > 2)
                    {
                        barValue = 2;
                    }
                    barValue *= BarUintHeight;
                    marginh = -barValue;
                }
                if (barValue < 0)
                {
                    if (barValue < -2)
                    {
                        barValue = 2;
                    }
                    barValue *= BarUintHeight;
                    marginh = barValue;
                }


                Grid g = new Grid();
                //g.Margin = new Thickness(0, marginh, 0, 0);

                Rectangle r = new Rectangle();
                r.Width = BarWidth;
                r.Height = barValue;
                r.VerticalAlignment = VerticalAlignment.Center;

                r.Fill = new SolidColorBrush(Color.FromRgb(255, 0, 0));

                g.Children.Add(r);
                g.Children.Add(new TextBlock()
                {
                    Text = Convert.ToString(Math.Round(value, 4))
                });


                BarContainer.Children.Add(g);
                BarContainer.InvalidateArrange();
            }
        }
    }
}
