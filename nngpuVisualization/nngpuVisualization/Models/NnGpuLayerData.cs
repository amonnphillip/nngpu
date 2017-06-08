using System;
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

        public double GetLargestDataValue()
        {
            double value = double.MinValue;
            for (int index = 0; index < data.Length; index++)
            {
                if (data[index] > value)
                {
                    value = data[index];
                }
            }

            return value;
        }

        public double GetSmallestDataValue()
        {
            double value = double.MaxValue;
            for (int index = 0; index < data.Length; index++)
            {
                if (data[index] < value)
                {
                    value = data[index];
                }
            }

            return value;
        }

        private double GetImageScale(out double floor)
        {
            double scale = 1;
            floor = double.MaxValue;
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

            return scale;
        }

        public byte[] ScaleImageData()
        {
            double floor;
            double scale = GetImageScale(out floor);

            byte[] imageData = new byte[width * height * depth * 4];
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

            return imageData;
        }

        public byte[] ScaleDepthImageData(int imageWidth, int imageHeight, int imageDepth)
        {
            double floor;
            double scale = GetImageScale(out floor);

            byte[] imageData = new byte[width * height * depth * 4];
            int i = 0;

            for (int y = 0; y < imageHeight; y++)
            {
                for (int x = 0; x < imageWidth; x++)
                {
                    for (int d = 0; d < imageDepth; d++)
                    {
                        int index = ((y * imageWidth * imageDepth) + (d * imageWidth) + x) * 4;
                        byte c = (byte)((data[i] - floor) * scale);
                        if (c > 255)
                        {
                            c = 255;
                        }

                        imageData[index] = c;
                        imageData[index + 1] = c;
                        imageData[index + 2] = c;
                        imageData[index + 3] = 0xff;

                        i++;
                    }
                }
            }

            return imageData;
        }

        public BitmapSource ByteArrayBitmapSource(byte[] imageData, int bitmapWidth, int bitmapHeight)
        {
            using (Bitmap bitmap = new Bitmap(bitmapWidth, bitmapHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb))
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

        public BitmapSource ToDepthImage()
        {
            byte[] imageData = ScaleDepthImageData(width, height, depth);
            return ByteArrayBitmapSource(imageData, width * depth, height);
        }

        public BitmapSource ToImage()
        {
            byte[] imageData = ScaleImageData();
            return ByteArrayBitmapSource(imageData, width, height);
        }

        public double Sum()
        {
            double sum = 0;
            for (int i = 0;i < data.Length;i++)
            {
                sum += data[i];
            }

            return sum;
        }
    }
}