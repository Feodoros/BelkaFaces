using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using csmatio;
using csmatio.io;
using csmatio.types;
using Emgu.CV.Dnn;

namespace WiderFacesNET
{
    class Program
    {
        static void Main(string[] args)
        {

            
            string pathMat = Environment.CurrentDirectory + "\wider_face_train.mat";
            string pathImages = Environment.CurrentDirectory + "\WIDER_train\\All";
            string modelFile = Environment.CurrentDirectory + "\opencv_face_detector_uint8.pb";
            string configFile = Environment.CurrentDirectory + "\opencv_face_detector.pbtxt";

            string file = pathImages + "\0_Parade_marchingband_1_1038.jpg";
            
            
            WiderFaces wf = new WiderFaces(pathImages, pathMat);

            List<int[][]> realBoxes = wf.GetBoxesEasy();
            List<string> imagesPaths = wf.GetPathsEasy();
            
            SSD ssd = new SSD(modelFile, configFile);

            List<int[][]> predictedFaces = ssd.DetectFacesSDD(imagesPaths);
            
            TestSystem testSystem = new TestSystem(realBoxes, predictedFaces);

            var metrics = testSystem.Metrics();
            
            Console.WriteLine();
            
            // 480, 212, 102, 120
        }
        
    }
}
