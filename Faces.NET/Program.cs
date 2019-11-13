using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using FaceRecognitionDotNet;
using DlibDotNet;
using Dlib = DlibDotNet.Dlib;

namespace faces
{
    class Program
    {
        private const string fileName = @"\recog1.jpg";
        private const string outputName = @"\output.jpg";

        static void Main(string[] args)
        {
            
            /// FaceDetectionWith_API
            Location[] coord = TestImage(fileName, Model.Hog);


            /// Face DetectionWith_DLIB
            using (var fd = Dlib.GetFrontalFaceDetector())
            {
                var img = Dlib.LoadImage<RgbPixel>(fileName);

                // find all faces in the image
                var faces = fd.Operator(img);
                foreach (var face in faces)
                {
                    // draw a rectangle for each face
                    Dlib.DrawRectangle(img, face, color: new RgbPixel(0, 255, 255), thickness: 4);
                }

                Dlib.SaveJpeg(img, outputName);
            }


            // The first thing we are going to do is load all our models.  First, since we need to
            // find faces in the image we will need a face detector:
            using (var detector = Dlib.GetFrontalFaceDetector())
            // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            // And finally we load the DNN responsible for face recognition.
            using (var net = DlibDotNet.Dnn.LossMetric.Deserialize("dlib_face_recognition_resnet_model_v1.dat"))

            using (var img = Dlib.LoadImageAsMatrix<RgbPixel>(fileName))
                
            using (var win = new ImageWindow(img))
            {
                var faces = new List<Matrix<RgbPixel>>();
                foreach (var face in detector.Operator(img))
                {
                    var shape = sp.Detect(img, face);
                    var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                    var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);

                    //faces.Add(move(face_chip));
                    faces.Add(faceChip);
                    
                    win.AddOverlay(face);
                }

                if (!faces.Any())
                {
                    Console.WriteLine("No faces found in image!");
                    return;
                }

                // This call asks the DNN to convert each face image in faces into a 128D vector.
                // In this 128D vector space, images from the same person will be close to each other
                // but vectors from different people will be far apart.  So we can use these vectors to
                // identify if a pair of images are from the same person or from different people.  
                var faceDescriptors = net.Operator(faces);

                // In particular, one simple thing we can do is face clustering.  This next bit of code
                // creates a graph of connected faces and then uses the Chinese whispers graph clustering
                // algorithm to identify how many people there are and which faces belong to whom.
                var edges = new List<SamplePair>();
                for (uint i = 0; i < faceDescriptors.Count; ++i)
                {
                    for (var j = i; j < faceDescriptors.Count; ++j)
                    {
                        // Faces are connected in the graph if they are close enough.  Here we check if
                        // the distance between two face descriptors is less than 0.6, which is the
                        // decision threshold the network was trained to use.  Although you can
                        // certainly use any other threshold you find useful.
                        var diff = faceDescriptors[i] - faceDescriptors[j];
                        if (Dlib.Length(diff) < 0.6)
                            edges.Add(new SamplePair(i, j));
                    }
                }

                Dlib.ChineseWhispers(edges, 100, out var numClusters, out var labels);

                // This will correctly indicate that there are 4 people in the image.
                Console.WriteLine($"number of people found in the image: {numClusters}");


                // Отобразим результат в ImageList
                var winClusters = new List<ImageWindow>();
                for (var i = 0; i < numClusters; i++)
                    winClusters.Add(new ImageWindow());
                var tileImages = new List<Matrix<RgbPixel>>();
                for (var clusterId = 0ul; clusterId < numClusters; ++clusterId)
                {
                    var temp = new List<Matrix<RgbPixel>>();
                    for (var j = 0; j < labels.Length; ++j)
                    {
                        if (clusterId == labels[j])
                            temp.Add(faces[j]);
                    }

                    winClusters[(int)clusterId].Title = $"face cluster {clusterId}";
                    var tileImage = Dlib.TileImages(temp);
                    tileImages.Add(tileImage);
                    winClusters[(int)clusterId].SetImage(tileImage);
                }


                // Finally, let's print one of the face descriptors to the screen.
                using (var trans = Dlib.Trans(faceDescriptors[0]))
                {
                    Console.WriteLine($"face descriptor for one face: {trans}");

                    // It should also be noted that face recognition accuracy can be improved if jittering
                    // is used when creating face descriptors.  In particular, to get 99.38% on the LFW
                    // benchmark you need to use the jitter_image() routine to compute the descriptors,
                    // like so:
                    var jitterImages = JitterImage(faces[0]).ToArray();
                    var ret = net.Operator(jitterImages);
                    using (var m = Dlib.Mat(ret))
                    using (var faceDescriptor = Dlib.Mean<float>(m))
                    using (var t = Dlib.Trans(faceDescriptor))
                    {
                        Console.WriteLine($"jittered face descriptor for one face: {t}");

                        // If you use the model without jittering, as we did when clustering the bald guys, it
                        // gets an accuracy of 99.13% on the LFW benchmark.  So jittering makes the whole
                        // procedure a little more accurate but makes face descriptor calculation slower.

                        Console.WriteLine("hit enter to terminate");
                        Console.ReadKey();

                        foreach (var jitterImage in jitterImages)
                            jitterImage.Dispose();

                        foreach (var tileImage in tileImages)
                            tileImage.Dispose();

                        foreach (var edge in edges)
                            edge.Dispose();

                        foreach (var descriptor in faceDescriptors)
                            descriptor.Dispose();

                        foreach (var face in faces)
                            face.Dispose();
                    }
                }

            }

            System.Console.ReadLine();

        }

        private static IEnumerable<Matrix<RgbPixel>> JitterImage(Matrix<RgbPixel> img)
        {
            // All this function does is make 100 copies of img, all slightly jittered by being
            // zoomed, rotated, and translated a little bit differently. They are also randomly
            // mirrored left to right.
            var rnd = new Rand();

            var crops = new List<Matrix<RgbPixel>>();
            for (var i = 0; i < 100; ++i)
                crops.Add(Dlib.JitterImage(img, rnd));

            return crops;
        }

        private static FaceRecognition _FaceRecognition;

        private static Location[] TestImage(string imageToCheck, Model model)
        {
            var directory = Path.GetFullPath("models");
            if (!Directory.Exists(directory))
            {
                Console.WriteLine($"Please check whether model directory '{directory}' exists");
            }

            _FaceRecognition = FaceRecognition.Create(directory);
            using (var unknownImage = FaceRecognition.LoadImageFile(imageToCheck))
            {
                var faceLocations = _FaceRecognition.FaceLocations(unknownImage, 0, model).ToArray();

                foreach (var faceLocation in faceLocations)

                    Console.WriteLine(string.Format("On image {0} we can see the face with coord: \n Bottom {1}, Left {3}, Right {2}, Top {4} ",
                        imageToCheck, faceLocation.Bottom, faceLocation.Left, faceLocation.Right, faceLocation.Top));
                return faceLocations;
            }    
        }
    }
}
