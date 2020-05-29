using System.Collections.Generic;
using csmatio.io;
using csmatio.types;
using Emgu.CV.Bioinspired;

namespace WiderFacesNET
{
    public class WiderFaces
    {
        
        private static string _pathMat;
        private readonly string _pathImages;
        private readonly MLCell _allFiles;
        private readonly MLCell _allBoxes;
        
        public WiderFaces(string pathImages, string pathMat)
        {
            WiderFaces._pathMat = pathMat;
            this._pathImages = pathImages;
            var mat = new MatFileReader(_pathMat);
            _allFiles = mat.Content["file_list"] as MLCell;
            _allBoxes = mat.Content["face_bbx_list"] as MLCell;
        }

        // Easy folders
        private List<string> _easyNames = new List<string>()
        {"Gymnastics", "Handshaking", "Waiter", "Conference",
            "Worker", "Parachutist", "Coach", "Meeting",
            "Aerobics", "Boat", "Dancing", "Swimming", 
            "Family", "Balloonist", "Dresses", "Couple", 
            "Jockey", "Tennis", "Spa", "Surgeons"};
            
        // Numbers of easy folders
        private readonly List<int> _easyDirectories = new List<int>()
        {
            35, 1, 25, 60, 26, 43, 51, 3, 39, 38, 34, 36, 13, 40, 47, 11, 41, 32, 20, 24
        };
        
        
        // Получаем пути к картинкам из легкой части датасета по
        // Параметру Scale
        public List<string> GetPathsEasy()
        {
            List<string> paths = new List<string>(){};

            foreach (var directory in _easyDirectories)
            {
                int count = ((MLCell) _allFiles.Cells[directory]).Size;
                for (int i = 0; i < count; i++)
                {
                    string fileString = ((MLChar) ((MLCell) _allFiles.Cells[directory]).Cells[i]).ContentToString(); 
                    string fileName = fileString.Split('\'')[1]; // File name
                    paths.Add($"{_pathImages}\\{fileName}.jpg");
                }
            }
            return paths;
        }
        
        // Получаем координаты лиц к картинкам из легкой части датасета по
        // Параметру Scale
        public List<int[][]> GetBoxesEasy()
        {
            List<int[][]> boxes = new List<int[][]>(){};
            
            foreach (var directory in _easyDirectories)
            {
                int count = ((MLCell) _allBoxes.Cells[directory]).Size;
                for (int i = 0; i < count; i++)
                {
                    // Boxes of image
                    int[][] imageBoxes = ((MLInt32) ((MLCell) _allBoxes.Cells[directory]).Cells[i]).GetArray();
                    boxes.Add(imageBoxes);
                }
            }
            return boxes;
        }
        
    }
}