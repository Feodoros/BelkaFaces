using System;
using System.Collections.Generic;

namespace WiderFacesNET
{
    public class TestSystem
    {
        private List<int[][]> _realFaces;
        private List<int[][]> _predictedFaces;

        public TestSystem(List<int[][]> realFaces, List<int[][]> predictedFaces)
        {
            this._realFaces = realFaces; 
            this._predictedFaces = predictedFaces;
        }

        // Посчитаем метрику IoU (Intersection over Union)
        public float IoU(int[] groundTruth, int[] predicted)
        {
            int[] boxA = 
            {   groundTruth[0], groundTruth[1], 
                groundTruth[0] + groundTruth[2], groundTruth[1] + groundTruth[3]
                
            };

            int[] boxB =
            {
                predicted[0], predicted[1],
                predicted[0] + predicted[2], predicted[1] + predicted[3]
            };
            
            // (x, y) - координаты прямоугольника из пересечения
            int xA = Math.Max(boxA[0], boxB[0]);
            int yA = Math.Max(boxA[1], boxB[1]);
            int xB = Math.Min(boxA[2], boxB[2]);
            int yB = Math.Min(boxA[3], boxB[3]);
            
            // Площадь прямоугольника, образованного пересечением двух других
            int interArea = Math.Max(0, xB - xA + 1) * Math.Max(0, yB - yA + 1);
            
            // Общая площадь двух прямоугольников
            int boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
            int boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);
            
            // Intersection over union
            float iou = interArea / (float) (boxAArea + boxBArea - interArea);
            return iou;
        }

        // Посчитаем метрики и классы TP, FP, FN
        public Dictionary<string, float> Metrics(int beta=1)
        {
            Dictionary<string, float> dict = new Dictionary<string, float>(){};

            float TP = 0; // true-positive
            float FP = 0; // false-positive 
            float FN = 0; // false-negative 

            // Бежим по всем ground-truth группам лиц
            for (int i = 0; i < _realFaces.Count; i++)
            {
                float tp = 0;
                float fp = 0;
                float fn = 0;
                
                // Бежим по всем ground-truth лицам на изображении
                for (int j = 0; j < _realFaces[i].Length; j++)
                {
                    // Сравниваем с нашими результатами
                    foreach (var box in _predictedFaces[i])
                    {
                        if (IoU(_realFaces[i][j], box) >= 0.5)
                        {
                            tp += 1;
                        }
                    }

                    fp = _predictedFaces[i].Length - tp;
                    fn = _realFaces[i].Length - tp;
                }

                TP += tp;
                FP += fp;
                FN += fn;
            }

            float recall = TP / (TP + FN);
            float precision = TP / (TP + FP);
            float fScore = ((1 + beta * beta) * (precision * recall)) / ((beta * beta * precision) + recall);

            dict["TP"] = TP;
            dict["FP"] = FP;
            dict["FN"] = FN;
            dict["Recall"] = recall;
            dict["Precision"] = precision;
            dict["F-Score"] = fScore;
            
            return dict;
        }
    }
}