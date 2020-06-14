using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;

namespace WiderFacesNET
{
    public class SSD
    {
        private readonly string _modelFile;
        private readonly string _configFile;

        public SSD(string modelFile, string configFile)
        {
            this._modelFile = modelFile;
            this._configFile = configFile;
        }

        // Ищем лица по списку изображений (SSD)
        public List<int[][]> DetectFacesSDD(List<string> imagePaths)
        {
            List<int[][]> allFaces = new List<int[][]>(){};
            int count = 0;
            
            // Ищем лица для каждого изображения
            foreach (var file in imagePaths)
            {
                List<int[]> faces = new List<int[]>();
                int i = 0;
                using (Image<Bgr, byte> image = new Image<Bgr, byte>(file))
                {
                    int cols = image.Width;

                    int rows = image.Height;

                    Net net = DnnInvoke.ReadNetFromTensorflow(_modelFile, _configFile);

                    net.SetInput(DnnInvoke.BlobFromImage(image.Mat, 1, new System.Drawing.Size(300, 300), default(MCvScalar), false, false));

                    Mat mat = net.Forward();

                    float[,,,] flt = (float[,,,])mat.GetData();

                    for (int x = 0; x < flt.GetLength(2); x++)
                    {
                        if (flt[0, 0, x, 2] > 0.2)
                        {
                            int left = Convert.ToInt32(flt[0, 0, x, 3] * cols);
                            int top = Convert.ToInt32(flt[0, 0, x, 4] * rows);
                            int right = Convert.ToInt32(flt[0, 0, x, 5] * cols) - left;
                            int bottom = Convert.ToInt32(flt[0, 0, x, 6] * rows) - top;
                            
                            int[] face = new[] {left, top, right, bottom};
                            faces.Add(face);
                            i++;
                        }
                    }
                }
                
                allFaces.Add(faces.ToArray());
                Console.WriteLine(count);
                count++;
            }
            
            return allFaces;
        }
    }
}