using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;

namespace WiderFacesNET
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelFile = "models/opencv_face_detector_uint8.pb";
            string configFile = "models/opencv_face_detector.pbtxt";

            var net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromTensorflow(modelFile, configFile);

            Console.WriteLine();
        }
    }
}
