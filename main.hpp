#ifndef MAIN_HPP_VDJI1OUJ
#define MAIN_HPP_VDJI1OUJ

#include <iostream>
#include <locale>
#include <vector>
#include <fstream>
#include "opencv2/face/facerec.hpp"
#include "opencv2/opencv.hpp"

using std::string;
using std::vector;

class FaceRecognizer{
	public:
		string LABEL_FILE = "label_file.npy";
		FaceRecognizer(string folderPath,cv::Size newSize,string cascadeFile,int arithmetic);
		~FaceRecognizer();

		void FillData(vector<cv::Mat>& images, vector<int>& labels);
		void FetchModel();
		void PredictPhoto(string imgPath);
		void PredictCamera();

	private:
		string _folderPath;
		cv::Size _newSize;;
		string _cascadeFile;
		cv::Ptr<cv::face::FaceRecognizer> _recognizer = nullptr;
		int _arithmetic;
		cv::CascadeClassifier _faceCascade;

		const int EIGEN_FACE_KEYS = 4500;
		const int LBPH_FACE_KEYS = 100;
		const int FISHER_FACE_KEYS = 4500;

		string _local_model_file = "LBPH_model.yml";
		int _keys = LBPH_FACE_KEYS;
		std::vector<string> _labels_name;
};

#endif /* end of include guard: MAIN_HPP_VDJI1OUJ */
