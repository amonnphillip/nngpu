﻿using System;
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

            double[] layerDataForward = laterDataGroup.layerData[0].data;
            double[] layerDataBackward = laterDataGroup.layerData[1].data;

            Outputs.Insert(0, layerDataBackward);
            if (Outputs.Count > 100)
            {
                Outputs.RemoveAt(100);
            }

            if (double.IsNaN(BarContainer.Width) ||
                double.IsNaN(BarContainer.Height))
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

            Line baseline = new Line();
            baseline.X1 = 0;
            baseline.Y1 = BarContainer.Height / 2;
            baseline.X2 = BarContainer.Width;
            baseline.Y2 = BarContainer.Height / 2;
            baseline.Stroke = new SolidColorBrush(Color.FromRgb(0, 0, 0));
            baseline.StrokeThickness = 1;
            BarContainer.Children.Add(baseline);

            double lastPointX = 0;
            double lastPointY = 0;
            for (int index = 0; index < Outputs.Count; index++)
            {
                double dataPointX = (index + 1) * 50;
                double dataPointY = ((aves[index] * -1) * scale) + (BarContainer.Height / 2);

                if (index > 0)
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



            /*
            for(int index = 0;index < layerDataForward.Length;index ++)
            {
                double value = layerDataForward[index];

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
            }*/
        }
    }
}
