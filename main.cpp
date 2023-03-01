#include "main.hpp"
#include <getopt.h>

#define VERSION 1

int gArithmetic = 2; //Default use LBPHFaceRecognizer
std::string gDirectory = "./lfw";
std::string gFile = "";

void CheckArgument(int argc,char *argv[]){
	const char* short_opts = "hvA:D:f:";
	const option long_opts[] = {
		{"help", no_argument, nullptr, 'h'},
		{"version", no_argument, nullptr, 'v'},
		{"arithmetic", required_argument, nullptr, 'A'},
		{"directory", required_argument, nullptr, 'D'},
		{"file", required_argument, nullptr, 'f'},
		{nullptr, no_argument, nullptr, 0}
	};

	int opt;
	while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
		switch (opt) {
			case 'h':
				std::cout << "Usage: " << argv[0] << " [options] [--] argument-1 argument-2 \
					\n\n  Options:\
					\n  -h, --help                Display this message\
					\n  -v, --version             Display version\
					\n  -A, --arithmetic [1|2|3]  Usage target arithmetic 1:EigenFace , 2:LBPHFace ,3:FisherFace\
					\n  -D, --directory DIRNAME   Directory used of training data\
					\n  -f, --file FILE           File used of test image\
					\n";
				exit(0);
			case 'v':
				std::cout << "Version: " << VERSION << "\n";
				exit(0);
			case 'A':
				gArithmetic = atoi(optarg);
				break;
			case 'D':
				gDirectory = optarg;
				break;
			case 'f':
				gFile = optarg;
				break;
			default:
				std::cout << "Unknown option: " << opt << "\n";
				exit(1);
		}
	}
}

int main(int argc, char *argv[]){
	CheckArgument(argc,argv);
	if(gArithmetic != 1 && gArithmetic != 2 && gArithmetic != 3){
		std::cout << "Arithmetic format error. try '--help'" << std::endl;
		exit(1);
	}
	FaceRecognizer face(gDirectory,cv::Size(92,112),"haarcascade_frontalface_default.xml",gArithmetic);
	face.FetchModel();
	if(!gFile.empty())
		face.PredictPhoto(gFile);
	else
		face.PredictCamera();
	return 0;
}

FaceRecognizer::FaceRecognizer(string folderPath,cv::Size newSize,string cascadeFile,int arithmetic)
	:_folderPath(folderPath),_newSize(newSize), _cascadeFile(cascadeFile), _arithmetic(arithmetic){
		_faceCascade.load(cascadeFile);
	}

FaceRecognizer::~FaceRecognizer(){}

void FaceRecognizer::FillData(vector<cv::Mat>& images, vector<int>& labels) {
	int index = 0;
	for (const auto& entry : std::__fs::filesystem::recursive_directory_iterator(_folderPath)) {
		if (entry.is_regular_file()) {
			string file_path = entry.path().string();
			cv::Mat img = imread(file_path, cv::IMREAD_GRAYSCALE);
			if (img.empty())
				continue;
			vector<cv::Rect> faces;
			_faceCascade.detectMultiScale(img, faces, 1.1, 5);
			for (const auto& face : faces) {
				cv::Mat face_roi = img(face);
				if (face_roi.empty())
					continue;
				resize(face_roi, face_roi, _newSize);
				images.push_back(face_roi);
				labels.push_back(index);
				_labels_name.push_back(entry.path().filename().string());
				index++;
			}
		}
	}
	std::ofstream ofs(LABEL_FILE);
	for (const auto& lab : _labels_name)
		ofs << lab << std::endl;

	std::cout << "\033[33m Into -> " << __FUNCTION__ << "()\033[0m" << std::endl;
}

void FaceRecognizer::FetchModel(){
	switch (_arithmetic) {
		case 1:
			_recognizer = cv::face::EigenFaceRecognizer::create();
			_local_model_file = "Eigen_model.yml";
			_keys = EIGEN_FACE_KEYS;
			break;
		case 2:
			_recognizer = cv::face::LBPHFaceRecognizer::create();
			_local_model_file = "LBPH_model.yml";
			_keys = LBPH_FACE_KEYS;
			break;
		case 3:
			_recognizer = cv::face::FisherFaceRecognizer::create();
			_local_model_file = "Fisher_model.yml";
			_keys = FISHER_FACE_KEYS;
			break;
	}

	if (!std::__fs::filesystem::exists(_local_model_file)) {
		vector<cv::Mat> images;
		vector<int> labels;

		struct timespec start_time, end_time;
		std::cout << "Filling image ..." << std::endl;
		clock_gettime(CLOCK_REALTIME, &start_time);
		FillData(images,labels);
		clock_gettime(CLOCK_REALTIME, &end_time);
		std::cout << "Fill done" << std::endl;
		std::cout << "Number of seconds -> "  <<  end_time.tv_sec - start_time.tv_sec << std::endl;

		std::cout << "Training ..." << std::endl;
		clock_gettime(CLOCK_REALTIME, &start_time);
		_recognizer->train(images,labels);
		clock_gettime(CLOCK_REALTIME, &end_time);
		std::cout << "Number of seconds -> "  <<  end_time.tv_sec - start_time.tv_sec << std::endl;
		std::cout << "Train done" << std::endl;

		_recognizer->save(_local_model_file);
	}else{
		std::cout << "Reading local mode ..." << std::endl;
		_recognizer->read(_local_model_file);
	}

	std::ifstream f(FaceRecognizer::LABEL_FILE);
	if (f.is_open()) {
		std::string line;
		while (std::getline(f, line)) {
			_labels_name.push_back(line);
		}
		f.close();
	}
	std::cout << "\033[33m Into -> " << __FUNCTION__ << "()\033[0m" << std::endl;
}


void FaceRecognizer::PredictPhoto(string imgPath){
	cv::Mat predict_image = cv::imread(imgPath,cv::IMREAD_GRAYSCALE);
	vector<cv::Rect> faces;
	_faceCascade.detectMultiScale(predict_image, faces, 1.1, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	for (const auto& face : faces) {
		cv::Mat img = predict_image(face);
		if(img.empty())
			continue;
		resize(img, img, _newSize);
		std::cout << "Predicting ..." << std::endl;
		int label = -1;
		double confidence = 0.0;
		_recognizer->predict(img,label,confidence);
		std::cout << "Find to -> "  << _labels_name[label] << std::endl;
		std::cout << "Confidence -> "  <<  confidence << std::endl;
	}
	std::cout << "\033[33m Into -> " << __FUNCTION__ << "()\033[0m" << std::endl;
}

void FaceRecognizer::PredictCamera(){
	cv::VideoCapture cap(0);
	while(1){
		cv::Mat frame , gray;
		//Get frame
		cap.read(frame);
		//Get gray
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		vector<cv::Rect> faces;
		//Get faces
		_faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
		for (const auto& face : faces) {
			cv::Mat face_roi = gray(face);
			//resize(face_roi, face_roi, _newSize);
			//cv::namedWindow("Image");
			//cv::imshow("Image",face_roi);
			//cv::waitKey(1);
			int label = -1;
			double confidence = 0.0;
			_recognizer->predict(face_roi,label,confidence);
			//cv::putText(frame, std::to_string(confidence), cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
			if(confidence < _keys){
				string name = _labels_name[label];
				cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
				cv::putText(frame, name, cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
			}else{
				cv::rectangle(frame, face, cv::Scalar(0, 0, 255), 2);
				cv::putText(frame, "unknown", cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
			}
		}

		cv::imshow("Face recognition",frame);

		if(cv::waitKey(1) == 'q')
			break;
	}
	cap.release();
	cv::destroyAllWindows();
	std::cout << "\033[33m Into -> " << __FUNCTION__ << "()\033[0m" << std::endl;
}
