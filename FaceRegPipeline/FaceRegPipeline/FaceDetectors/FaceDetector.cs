using DlibDotNet;

namespace FaceRegPipeline.FaceDetectors
{
    public interface FaceDetector
    {
        Rectangle[] DetectFaces(string inputFile, string outputFile);
    }
}