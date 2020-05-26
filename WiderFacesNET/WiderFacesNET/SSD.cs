using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Dnn;

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

        public void DetectFacesSDD(List<string> imagePaths)
        {
            Net net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromTensorflow(_modelFile, _configFile);
            Mat image = CvInvoke.Imread(imagePaths[0]);
            var blob = DnnInvoke.BlobFromImage(image, 1, new Size(300, 300));
            
        }
    }
}