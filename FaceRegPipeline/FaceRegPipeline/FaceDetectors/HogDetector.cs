using DlibDotNet;

namespace FaceRegPipeline.FaceDetectors
{
    public class HogDetector : FaceDetector
    {
        private FrontalFaceDetector detector;
        
        public HogDetector()
        {
            detector = Dlib.GetFrontalFaceDetector();
        }
        
        public Rectangle[] DetectFaces(string inputFile, string outputFile)
        {
            var img = Dlib.LoadImage<RgbPixel>(inputFile);
            var faces = detector.Operator(img);
            foreach (var face in faces)
            {
                Dlib.DrawRectangle(img, face, new RgbPixel(0, 255, 255), 4);
            }

            Dlib.SaveJpeg(img, outputFile);
            return faces;
        }
    }
}