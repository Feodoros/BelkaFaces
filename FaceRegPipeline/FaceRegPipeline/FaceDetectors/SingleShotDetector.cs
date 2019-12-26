using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using DlibDotNet;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Rectangle = System.Drawing.Rectangle;

namespace FaceRegPipeline.FaceDetectors
{
    public class SingleShotDetector : FaceDetector
    {
        private static readonly string BasePath =
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..\\..\\..\\"));

        private static readonly string SsdPath =
            Path.Combine(BasePath, "models", "OpenCV");

        private static readonly string Proto = Path.Combine(SsdPath, "deploy.prototxt.txt");
        private static readonly string Model = Path.Combine(SsdPath, "res10_300x300_ssd_iter_140000.caffemodel");

        private static readonly string InputDirectory = Path.Combine(BasePath, "images");
        private static readonly string TestImage = Path.Combine(InputDirectory, "1.jpg");
        private static readonly string Output = Path.Combine(InputDirectory, "out.jpg");

        public DlibDotNet.Rectangle[] DetectFaces(string inputImage, string outputImage)
        {
            var net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromCaffe(Proto, Model);
            var image = new Emgu.CV.Image<Bgr, byte>(inputImage);

            var height = image.Height;
            var width = image.Width;

            var blob = Emgu.CV.Dnn.DnnInvoke.BlobFromImage(image.Resize(300, 300, Inter.Linear), 1.0,
                new Size(300, 300),
                new MCvScalar(104.0, 177.0, 123.0));

            net.SetInput(blob);
            var dets = net.Forward();
            var mat = (float[,,,]) dets.GetData();

            var rects = new List<DlibDotNet.Rectangle>();
            for (var i = 0; i < dets.SizeOfDimension[2]; i++)
            {
                var confidence = mat[0, 0, i, 2];
                if (confidence > 0.55)
                {
                    var startX = (int) (mat[0, 0, i, 3] * width);
                    var startY = (int) (mat[0, 0, i, 4] * height);
                    var endX = (int) (mat[0, 0, i, 5] * width);
                    var endY = (int) (mat[0, 0, i, 6] * height);

                    var rect = new Rectangle(startX, startY, endX - startX, endY - startY);
                    rects.Add(new DlibDotNet.Rectangle(startX, startY, endX, endY));
                    image.Draw(rect, new Bgr(Color.Red), 3);
                }
            }

            image.Save(outputImage);
            return rects.ToArray();
        }
    }
}