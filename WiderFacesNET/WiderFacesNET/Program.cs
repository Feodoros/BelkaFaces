using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using csmatio;
using csmatio.io;
using csmatio.types;

namespace WiderFacesNET
{
    class Program
    {
        static void Main(string[] args)
        {
            string pathMat = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\WiderFaces\wider_face_train.mat";
            string pathImages = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\WiredFaces\WIDER_train\All";
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

            
            MatFileReader mat = new MatFileReader(pathMat);
            
            MLCell allFiles = mat.Content["file_list"] as MLCell;
            MLCell allBoxes = mat.Content["face_bbx_list"] as MLCell;
            
            // Get faces of image
            MLCell x = allFiles.Cells[1] as MLCell;
            MLChar y =  x.Cells[1] as MLChar;
            var z = y.ContentToString(); // File name
            MLCell x1 = allBoxes.Cells[1] as MLCell;
            MLInt32 y1 = x1.Cells[1] as MLInt32;
            var z1 = y1.GetArray(); // Boxes of file
            
            // ReSharper disable once InvalidXmlDocComment
            /// TODO:
            /// Получить номера easy folders питоном
            /// Получить изображения + координаты из easy folders
            /// Распарсить имя файла
            /// метод IoU, calculate metrics
            /// SSD
            
            
            //  Get easy folders
            List<string> easyNames = new List<string>(){"Gymnastics", "Handshaking", "Waiter", "Conference",
                "Worker", "Parachutist", "Coach", "Meeting",
                "Aerobics", "Boat", "Dancing", "Swimming", 
                "Family", "Balloonist", "Dresses", "Couple", 
                "Jockey", "Tennis", "Spa", "Surgeons"};
            
            
            Console.WriteLine();
        }
    }
}
