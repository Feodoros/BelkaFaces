using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;

namespace WiderFacesNET
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelFile = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\Models\opencv_face_detector_uint8.pb";
            string configFile = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\Models\opencv_face_detector.pbtxt";
            string haar = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\Models\haarcascade_frontalface_default.xml";

            CascadeClassifier faceCascade = new CascadeClassifier(haar);

            string imagePath = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\testDataSet\musk.jpg";

            Mat img = CvInvoke.Imread(imagePath, Emgu.CV.CvEnum.ImreadModes.AnyColor);
            Image<Bgr, Byte> image = img.ToImage<Bgr, Byte>();
            Image<Gray, byte> grayframe = image.Convert<Gray, byte>();

            var net = Emgu.CV.Dnn.DnnInvoke.ReadNetFromTensorflow(modelFile, configFile);

      
            var faces = faceCascade.DetectMultiScale(grayframe, 1.1, 10,              
                  new System.Drawing.Size(20, 20));

            Console.WriteLine();
        }
    }
}
