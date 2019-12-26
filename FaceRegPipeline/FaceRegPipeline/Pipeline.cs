using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DlibDotNet;
using DlibDotNet.Dnn;
using FaceRegPipeline.FaceDetectors;

namespace FaceRegPipeline
{
    public class Pipeline
    {
        private static readonly string BasePath =
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..\\..\\..\\"));

        private static readonly string InputDirectory = Path.Combine(BasePath, "images");
        private static readonly string OutputDirectory = Path.Combine(BasePath, "output");

        private static readonly string ShapePredictorPath =
            Path.Combine(BasePath, "models", "shape_predictor_5_face_landmarks.dat");

        private static readonly string EmbPath =
            Path.Combine(BasePath, "models", "dlib_face_recognition_resnet_model_v1.dat");

        private static readonly IEnumerable<string> SupportedFormats = new[] {".jpeg", ".jpg", ".png"};

        static void Main()
        {
            var detector = new SingleShotDetector();
            ProcessImages(InputDirectory, OutputDirectory, ShapePredictorPath, EmbPath, detector);
        }

        private static void ProcessImages(string inputDir, string outputDir, string shapeDetectorPath,
            string embeddingModelPath, FaceDetector detector, double threshold = 0.6)
        {
            var files = Directory.GetFiles(inputDir);
            var images = files.Where(file => SupportedFormats.Any(file.EndsWith));

            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }
            
            using var predictor = ShapePredictor.Deserialize(shapeDetectorPath);
            using var dnn = LossMetric.Deserialize(embeddingModelPath);

            var dict = new Dictionary<Matrix<RgbPixel>, Matrix<float>>();
            foreach (var inputFile in images)
            {
                var img = Dlib.LoadImage<RgbPixel>(inputFile);

                var formatIdx = inputFile.LastIndexOf('.');
                var format = inputFile.Substring(formatIdx);

                var outputFile = inputFile.Remove(formatIdx).Split(Path.DirectorySeparatorChar).Last();
                outputFile = string.Concat(outputFile, "_result", format);
                outputFile = Path.GetFullPath(Path.Combine(outputDir, outputFile));

                var faces = detector.DetectFaces(inputFile, outputFile);
                var embeddings = GetFaceEmbeddings(img, faces, predictor, dnn);

                foreach (var chip in embeddings.Keys)
                    dict.Add(chip, embeddings[chip]);
            }

            RecognizeFaces(dict.Keys.ToList(), dict.Values.ToArray(), outputDir, threshold);
        }

        private static Dictionary<Matrix<RgbPixel>, Matrix<float>> GetFaceEmbeddings(Array2DBase img, Rectangle[] faces,
            ShapePredictor predictor, LossMetric dnn)
        {
            var chips = new List<Matrix<RgbPixel>>();
            foreach (var face in faces)
            {
                // detect landmarks
                var shape = predictor.Detect(img, face);

                // extract normalized and rotated 150x150 face chip
                var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);

                // convert the chip to a matrix and store
                var matrix = new Matrix<RgbPixel>(faceChip);
                chips.Add(matrix);
            }

            var embDict = new Dictionary<Matrix<RgbPixel>, Matrix<float>>();
            if (!chips.Any())
            {
                return embDict;
            }

            // put each face in a 128D embedding space
            var descriptors = dnn.Operator(chips);

            for (var i = 0; i < descriptors.Count; i++)
            {
                embDict.Add(chips[i], descriptors[i]);
            }

            return embDict;
        }

        private static void RecognizeFaces(List<Matrix<RgbPixel>> chips, Matrix<float>[] descriptors, string outputDir,
            double threshold = 0.6)
        {
            var edges = new List<SamplePair>();

            for (uint i = 0; i < descriptors.Length; ++i)
            for (var j = i; j < descriptors.Length; ++j)
                if (Dlib.Length(descriptors[i] - descriptors[j]) < threshold)
                    edges.Add(new SamplePair(i, j));

            Dlib.ChineseWhispers(edges, 100, out var clusters, out var labels);

            for (var i = 0; i < chips.Count; i++)
            {
                var outputClusterPath = Path.Combine(outputDir, $"Person {labels[i]}");
                if (!Directory.Exists(outputClusterPath))
                {
                    Directory.CreateDirectory(outputClusterPath);
                }

                Dlib.SaveJpeg(chips[i], Path.Combine(outputClusterPath, $"Face_{i}.jpeg"));
            }
        }
    }
}