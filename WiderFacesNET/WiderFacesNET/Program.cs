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
            string pathMat = "C:\\Users\\Fedor\\Documents\\Projects\\BelkaFaces\\BelkaFaces_Git\\WiderFaces\\wider_face_train.mat";
            string pathImages = "C:\\Users\\Fedor\\Documents\\Projects\\BelkaFaces\\WiredFaces\\WIDER_train\\All";
            string modelFile = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\Models\opencv_face_detector_uint8.pb";
            string configFile = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\Models\opencv_face_detector.pbtxt";
            string haar = @"C:\Users\Fedor\Documents\Projects\BelkaFaces\BelkaFaces_Git\Models\haarcascade_frontalface_default.xml";
            
            
            
            
            // ReSharper disable once InvalidXmlDocComment
            /// TODO:
            /// Получить номера easy folders питоном + 
            /// Получить изображения + координаты из easy folders
            /// Распарсить имя файла
            /// метод IoU, calculate metrics
            /// SSD
            
            WiderFaces wf = new WiderFaces(pathImages, pathMat);

            var x = wf.GetBoxesEasy();
            var y = wf.GetPathsEasy();
            
            Console.WriteLine();
        }
    }
}
