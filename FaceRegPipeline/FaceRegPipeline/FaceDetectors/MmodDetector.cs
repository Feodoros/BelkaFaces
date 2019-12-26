using System;
using System.Collections.Generic;
using System.IO;
using DlibDotNet;
using DlibDotNet.Dnn;

namespace FaceRegPipeline.FaceDetectors
{
    public class MmodDetector : FaceDetector
    {
        private static readonly string BasePath =
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..\\..\\..\\"));

        private static readonly string MmodPath =
            Path.Combine(BasePath, "models", "mmod_human_face_detector.dat");

        private LossMmod model;

        public MmodDetector()
        {
            model = LossMmod.Deserialize(MmodPath);
        }

        public Rectangle[] DetectFaces(string inputFile, string outputFile)
        {
            var img = Dlib.LoadImage<RgbPixel>(inputFile);
            var mat = new Matrix<RgbPixel>(img);
            var faces = new List<Rectangle>();

            using (var dets = model.Operator(mat))
            {
                foreach (var det in dets)
                {
                    foreach (var face in det)
                    {
                        Dlib.DrawRectangle(img, face.Rect, new RgbPixel(0, 255, 255), 4);
                        faces.Add(face.Rect);
                    }
                }

                Dlib.SaveJpeg(img, outputFile);
            }

            return faces.ToArray();
        }
    }
}