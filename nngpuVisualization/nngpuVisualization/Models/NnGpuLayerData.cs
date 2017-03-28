using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace nngpuVisualization
{
    public class NnGpuLayerData
    {
        public NnGpuLayerDataType type;
        public int width;
        public int height;
        public int depth;
        public double[] data;

        public BitmapSource ToImage()
        {
            byte[] imageData = new byte[width * height * 4];
            int i = 0;
            for (int x = 0; x < imageData.Length; x += 4)
            {
                byte c = (byte)(data[i] * 255);
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

            Bitmap bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0,
                                                bitmap.Width,
                                                bitmap.Height),
                                  ImageLockMode.WriteOnly,
                                  bitmap.PixelFormat);

            IntPtr pNative = bmpData.Scan0;
            Marshal.Copy(imageData, 0, pNative, imageData.Length);

            bitmap.UnlockBits(bmpData);

            var handle = bitmap.GetHbitmap();

            return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
        }


    }
}