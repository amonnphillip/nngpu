﻿using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media.Imaging;

namespace nngpuVisualization
{


    public class NnGpuLayerData
    {
        [DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);

        public NnGpuLayerDataType type;
        public int width;
        public int height;
        public int depth;
        public double[] data;

        public BitmapSource ToImage()
        {
            double scale = 1;
            double floor = double.MaxValue;
            double top = double.MinValue;
            for (int index = 0; index < data.Length; index++)
            {
                if (data[index] < floor)
                {
                    floor = data[index];
                }

                if (data[index] > top)
                {
                    top = data[index];
                }
            }

            scale = 255 / (top - floor);

            byte[] imageData = new byte[width * height * 4];
            int i = 0;
            for (int x = 0; x < imageData.Length; x += 4)
            {
                byte c = (byte)((data[i] - floor) * scale);
                if (c > 255)
                {
                    c = 255;
                }

                imageData[x] = c;
                imageData[x + 1] = c;
                imageData[x + 2] = c;
                imageData[x + 3] = 0xff;

                i++;
            }

            using (Bitmap bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb))
            { 
                BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0,
                                                    bitmap.Width,
                                                    bitmap.Height),
                                      ImageLockMode.WriteOnly,
                                      bitmap.PixelFormat);

                IntPtr pNative = bmpData.Scan0;
                Marshal.Copy(imageData, 0, pNative, imageData.Length);

                bitmap.UnlockBits(bmpData);

                IntPtr handle = bitmap.GetHbitmap();

                BitmapSource bitmapSource = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());

                DeleteObject(handle);

                return bitmapSource;
            }
        }


    }
}