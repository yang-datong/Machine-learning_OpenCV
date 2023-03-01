机器学习：是让计算机通过学习数据规律，自动进行分类、预测或者决策的一种方法。这种方法需要人工指定计算机需要学习的特征，然后通过训练算法去调整模型参数，使得模型能够准确地对新数据进行分类或预测。

深度学习：是机器学习的一种，其核心是利用神经网络来提取特征和学习模型。相对于传统的机器学习方法，深度学习不需要手工设计特征，而是利用多层次的神经网络自动学习特征，并通过大量的训练数据来调整神经网络的权重，从而实现对新数据的分类、预测等任务。

简单来说，机器学习是通过训练算法去让计算机学习数据规律，而深度学习则是通过神经网络自动提取特征并学习模型来完成同样的任务。



## 安装（python）

`opencv-python`是OpenCV的Python绑定，可以让Python开发者使用OpenCV库的功能。这个库包含了OpenCV的核心模块和一些常用的扩展模块，比如imgproc（图像处理）、videoio（视频输入/输出）、highgui（图形界面）等。

`opencv-contrib-python`是对`opencv-python`的扩展，包含了一些不在`opencv-python`中的额外模块，比如face（人脸识别）、text（OCR识别）、xfeatures2d（SIFT/SURF特征提取器）等。这些额外的模块都是由OpenCV社区贡献的，但不是所有模块都被官方OpenCV库接受并集成。

因此，如果您只需要使用OpenCV的核心模块和一些常用的扩展模块，那么`opencv-python`就足够了；如果您需要使用更多的功能，比如人脸识别、OCR等，那么您需要使用`opencv-contrib-python`。

```bash
pip install opencv-contrib-python==4.7.0.72 numpy
```

## 使用

下载模型数据

```bash
wget https://img-blog.csdnimg.cn/20210306184517460.jpg#pic_center -O 1.jpg
wget https://img-blog.csdnimg.cn/20210306184536600.jpg#pic_center  -O 2.jpg
wget https://img-blog.csdnimg.cn/20210306184551606.jpg#pic_center  -O test.jpg
```

### Eigen、LBPH、Fisher算法

1. Eigenfaces算法：

原理：使用主成分分析（PCA）对人脸图像进行降维，然后使用k最近邻算法进行分类。

优点：算法简单易懂，训练速度快，识别准确率高。

缺点：对光照、姿态等因素敏感，需要对输入图像进行预处理，否则会影响识别准确率。

2. LBPH算法：

原理：将人脸图像划分成不同的小块，对每个小块提取局部二值模式（LBP）特征，然后将所有小块的特征拼接成一个向量，使用k最近邻算法进行分类。

优点：对光照、姿态等因素具有较好的鲁棒性，不需要对输入图像进行预处理。

缺点：在处理复杂表情、遮挡等情况时，识别准确率较低。

3. Fisherfaces算法：

原理：使用线性判别分析（LDA）对人脸图像进行降维，然后使用k最近邻算法进行分类。

优点：对光照、姿态等因素具有较好的鲁棒性，与PCA相比，更加注重类间差异，识别准确率较高。

缺点：算法复杂度较高，需要较长的训练时间和更多的计算资源。

**综上所述，选择何种算法需要根据具体应用场景进行考虑。**

- 如果需要快速训练和较高的识别准确率，可以选择Eigenfaces算法；
- 如果对光照、姿态等因素具有较好的鲁棒性，可以选择LBPH算法；
- 如果对识别准确率有更高的要求，并且拥有更多的计算资源，可以选择Fisherfaces算法。

```python
#!/usr/local/bin/python3.9
import cv2
import numpy as np

# 读取用于训练模型的数据并将其转为灰度图像
images = [cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE),cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)]

#设置训练标签，几张人脸就几个标签，相当于起id
labels = [0, 1]

arithmetic = input(
'''
Chose Arithmetic realize :
1.Eigenfaces算法模型
2.LBPH算法模型
3.Fisher算法模型
'''
)

if arithmetic == "1":
    #创建Eigenfaces算法模型
    recognizer = cv2.face.EigenFaceRecognizer_create()
elif arithmetic == "2":
    #创建LBPH算法模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    #创建Fisher算法模型
    recognizer = cv2.face.FisherFaceRecognizer_create()


print("train...")
#将训练图像和标签传入进行模型训练,这里是同步的
recognizer.train(images, np.array(labels))
print("train ok ")

#加载一张待识别的人脸图像，并将其转化为灰度图像
predict_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

print("predict...")
#对待识别图像进行人脸识别，并返回预测结果(label)和置信度(confidence)
label, confidence = recognizer.predict(predict_image)

if label == 0:
    print("匹配的人脸为尼根")
elif label == 1:
    print("匹配的人脸为瑞克")
print("confidence=", confidence)
```

在LBPH算法中，可信度值是表示预测结果的置信度或准确性的一个浮点数。其值越小表示预测结果越可信，也就是说预测结果越接近测试图像。这个值的计算通常基于算法内部的分类器和距离度量方式，不同的实现可能会有不同的计算方式。通常情况下，可信度值越小，表示预测结果越可靠，但具体的取值和含义也需要根据具体算法和数据集进行分析和理解。同时数据标签也必须是整数

EigenFaces人脸识别唯一的缺陷就是不管是训练的图像，还是测试的图像，其大小必须一致。而LBPH人脸识别并不需要图像大小一致。还有EigenFaces人脸识别返回的confidence大小介于0到20000，只要低于5000都被认为是可靠的结果。

### 模型保存与模型加载

因为模型训练是一个非常耗时的过程，并且相同的数据每次训练出来的模型都是一样，那么就将训练好的模型保存在本地，以便后续使用

```python
#!/usr/local/bin/python3.9
import cv2
import numpy as np
import os

#封装模型使用调用
def open_moduel(images,labels):
    #创建LBPH算法模型
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists("LBPH_model.yml"):
        print("train...")
        #将训练图像和标签传入进行模型训练,这里是同步的
        recognizer.train(images, np.array(labels))
        print("train ok ")
        #保存训练完成的模型
        recognizer.save("LBPH_model.yml")
    else:
        #获取本地已经训练完成的模型
        recognizer.read("LBPH_model.yml")

def main():
    # 读取用于训练模型的数据并将其转为灰度图像
    images = [cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE),cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)]

    #设置训练标签，几张人脸就几个标签，相当于起id
    labels = [0, 1]

    #封装模型使用调用
    open_moduel(images,labels)

    #加载一张待识别的人脸图像，并将其转化为灰度图像
    predict_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

    print("predict...")
    #对待识别图像进行人脸识别，并返回预测结果(label)和置信度(confidence)
    label, confidence = recognizer.predict(predict_image)

    if label == 0:
        print("匹配的人脸为尼根")
    elif label == 1:
        print("匹配的人脸为瑞克")
    print("confidence=", confidence)

main()
```

### 训练大量的数据集

当前目录文件结构：

.
├── LBPH_model.yml
├── facedecetion.py
├── haarcascade_frontalface_default.xml
├── label_file.npy
└── lfw

```python
#!/usr/local/bin/python3.9
import cv2
import numpy as np
import os
import sys

if len(sys.argv) == 1:
    print("need test image . such as -> "+sys.argv[0] + " test.jpg")
    exit(0)

# wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
folder_path = "./lfw"
new_size = (250, 250)


def fill_data(images, labels, labels_name):
    print("fill data")
    # 获取文件夹内所有文件的路径
    i = 0
    # 遍历文件夹及其所有子文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            img = cv2.resize(img, new_size)
            images.append(img)
            labels.append(i)
            labels_name.append(os.path.basename(file_path))
            i += 1
    with open("label_file.npy", 'w') as f:
        for lab in labels_name:
            f.write(lab+"\n")


# 封装模型使用调用
def open_model():
    # 创建LBPH算法模型
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists("LBPH_model.yml"):
        images = []
        labels = []
        labels_name = []
        fill_data(images, labels, labels_name)

        print("train...")
        # 将训练图像和标签传入进行模型训练,这里是同步的
        recognizer.train(images, np.array(labels))
        print("train ok ")
        # 保存训练完成的模型
        recognizer.save("LBPH_model.yml")
    else:
        print("read local model ...")
        # 获取本地已经训练完成的模型
        recognizer.read("LBPH_model.yml")


def main():
    # 封装模型使用调用
    open_model()

    # 加载一张待识别的人脸图像，并将其转化为灰度图像
    predict_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    print("predict...")
    # 对待识别图像进行人脸识别，并返回预测结果(label)和置信度(confidence)
    label, confidence = recognizer.predict(predict_image)
    with open("./label_file.npy") as f:
        labels_name = f.readlines()

    print("find -> ", labels_name[label], end="")
    print("confidence -> ", confidence)

main()
```

但是上面训练的是整张图片，并且识别的时候也是整张照片进行识别的，如果图片中非人物信息非常多，那么就会影响识别成功率，我们在训练模型和识别图片的时候再给图片加上一个人脸检测框，只对人脸部分进行裁剪训练

```python
#!/usr/local/bin/python3.9
import cv2
import numpy as np
import os
import sys

if len(sys.argv) == 1:
    print("need test image . such as -> "+sys.argv[0] + " test.jpg")
    exit(0)

# wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
folder_path = "./lfw"
new_size = (92, 112)


def fill_data(images, labels, labels_name):
    print("fill data")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 获取文件夹内所有文件的路径
    i = 0
    # 遍历文件夹及其所有子文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            faces = face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                img = img[y:y+h, x:x+w]
                if img is None or img.size == 0:
                    continue
                img = cv2.resize(img, new_size)
                # cv2.imshow('Image', img)
                # cv2.waitKey(1)
                images.append(img)  # 调整尺寸为 92x112
                labels.append(i)
                labels_name.append(os.path.basename(file_path))
                i += 1

    with open("label_file.npy", 'w') as f:
        for lab in labels_name:
            f.write(lab+"\n")


# 封装模型使用调用
def open_model():
    # 创建LBPH算法模型
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists("LBPH_model.yml"):
        images = []
        labels = []
        labels_name = []
        fill_data(images, labels, labels_name)

        print("train...")
        # 将训练图像和标签传入进行模型训练,这里是同步的
        recognizer.train(images, np.array(labels))
        print("train ok ")
        # 保存训练完成的模型
        recognizer.save("LBPH_model.yml")
    else:
        print("read local model ...")
        # 获取本地已经训练完成的模型
        recognizer.read("LBPH_model.yml")


def main():
    # 封装模型使用调用
    open_model()

    # 加载一张待识别的人脸图像，并将其转化为灰度图像
    predict_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        predict_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        img = predict_image[y:y+h, x:x+w]
        print("predict...")
        label, confidence = recognizer.predict(img)
        # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.putText(img, "Label: {}, Confidence: {}".format(label, confidence), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    with open("./label_file.npy") as f:
        labels_name = f.readlines()

    print("find -> ", labels_name[label], end="")
    print("confidence -> ", confidence)

main()
```

那么再添加一个相机识别功能就大功造成了，为了方便管理，这个改写成面向对象代码结构，函数逻辑并没有改变：

```python
#!/usr/local/bin/python3.9
import cv2
import numpy as np
import os
import sys


class FaceRecognizer:
    LABEL_FILE = "label_file.npy"

    def __init__(self, folder_path, new_size=(92, 112), cascade_file='haarcascade_frontalface_default.xml',arithmetic="2"):
        self.folder_path = folder_path
        self.new_size = new_size
        self.cascade_file = cascade_file
        self.recognizer = None
        self.labels_name = []
        self.arithmetic = arithmetic
        self.local_model_file = "LBPH_model.yml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_file)

    def use_hard(self):
        pass

    def fill_data(self, images, labels):
        print("fill data ...")
        index = 0
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                faces = self.face_cascade.detectMultiScale(
                    img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    img = img[y:y+h, x:x+w]
                    if img is None or img.size == 0:
                        continue
                    img = cv2.resize(img, self.new_size)
                    images.append(img)
                    labels.append(index)
                    self.labels_name.append(os.path.basename(file_path))
                    index += 1

        with open(FaceRecognizer.LABEL_FILE, 'w') as f:
            for lab in self.labels_name:
                f.write(lab+"\n")

    def fetch_model(self):
        if self.arithmetic == "1":
            self.recognizer = cv2.face.EigenFaceRecognizer_create()
            self.local_model_file = "Eigen_model.yml"
        elif self.arithmetic == "2":
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.local_model_file = "LBPH_model.yml"
        elif self.arithmetic == "3":
            self.recognizer = cv2.face.FisherFaceRecognizer_create()
            self.local_model_file = "Fisher_model.yml"

        if not os.path.exists(self.local_model_file):
            images = []
            labels = []
            self.fill_data(images, labels)

            print("train...")
            self.recognizer.train(images, np.array(labels))
            print("train ok ")
            self.recognizer.save(self.local_model_file)
        else:
            print("read local model ...")
            self.recognizer.read(self.local_model_file)

        with open(FaceRecognizer.LABEL_FILE) as f:
            self.labels_name = f.readlines()

    def predict_photo(self, img_path):
        predict_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces = self.face_cascade.detectMultiScale(
            predict_image, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            img = predict_image[y:y+h, x:x+w]
            print("predict...")
            label, confidence = self.recognizer.predict(img)

        print("find -> ", self.labels_name[label], end="")
        print("confidence -> ", confidence)

    def predict_camera(self):
        # 打开摄像头，进行人脸识别
        cap = cv2.VideoCapture(0)
        while True:
            # 读取摄像头中的图像
            ret, frame = cap.read()
            # 将图像转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 检测人脸并进行识别
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                # 获取人脸区域并进行预测
                face_roi = gray[y:y+h, x:x+w]
                label, confidence = self.recognizer.predict(face_roi)
                # 绘制矩形框和标签
                if confidence < 100:
                    name = label
                    color = (0, 255, 0)
                else:
                    name = 'unknown'
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(frame, str(
                    self.labels_name[label]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)

            # 显示识别结果
            cv2.imshow('face recognition', frame)

            # 按下q键退出
            if cv2.waitKey(1) == ord('q'):
                break

        # 释放摄像头并关闭窗口
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    #arithmetic 1 -> Eigen_model.yml
    #arithmetic 2 -> LBPH_model.yml
    #arithmetic 3 -> Fisher_model.yml
    face = FaceRecognizer("./lfw",arithmetic="2")
    face.fetch_model()
    if len(sys.argv) == 1:
        face.predict_camera()
    else:
        face.predict_photo(sys.argv[1])
```

### 对应的C++版本

```c++
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
```



```c++
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
```









# Mac安装Open CV库

```bash
brew install opencv
```

```bash
#依赖项为：
Installing dependencies for opencv: eigen, flags, glog, metis, isl, mpfr, libmpc, 1z4, xz, zstd, gcc, openblas, suite-sparse, tbb, ceres-solver, highway, imath, jpeg-turbo, libpng, libtiff, little-cms2, webp,jpeg-xl, aom, aribb24, davld, freetype, fontconfig, ca-certificates,
1ibunistring, gettext, libidn2, libtasn1, nettle, p11-kit, openssl@1.1,libnghttp2, unbound, gnutls, pcre2, glib, xorgproto, libxau, libxdmcp,libxcb, libx11, libxext, libxrender, pixman, icu4c, harfbuzz, libunibreak, libass, libbluray, mbedtls, librist, libvpx, opencore-amr, ravle,ibsamplerate, flac, mpg123, libsndfile, rubberband, sdl2, speex, srt,Svt-avl, leptonica, libarchive, pango, tesseract, ffmpeg, numpy, protobuf, readline, sqlite, python@3.11, boost, double-conversion, gl2ps, glewlibaec, hdf5, jsoncpp, netcdf, pugixml, qt@5, pyqt@5, utf&cpp and vtk
```

这个依赖项目很恐怖了，大半个音视频圈子的工具

因为我安装的是`opencv4`所以目录头文件在`/usr/local/Cellar/opencv/4.7.0_1/include/opencv4/`下面默认情况下brew会自动建立`include`目录下的软链接，但是我多了一层所以我编译的时候采用的手动链接，或者可以直接将`opencv4/`下面的`opencv2/`移动到`/usr/local/include`下面就可以直接引入头文件了

```c++
#include "opencv2/core/cvstd_wrapper.hpp"
#include "opencv2/face/facerec.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <vector>

int main(){
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	images.push_back(cv::imread("1.jpg",cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread("2.jpg",cv::IMREAD_GRAYSCALE));
	labels.push_back(0);
	labels.push_back(1);

	int arithmetic;
	std::cout << "Choose an arithmetic realization (1-3):\n"
					 << "1. Eigenfaces algorithm model\n"
					 << "2. LBPH algorithm model\n"
					 << "3. Fisher algorithm model\n";
	std::cin >> arithmetic;

	cv::Ptr<cv::face::FaceRecognizer> recognizer = nullptr;
	if(arithmetic == 1){
		recognizer = cv::face::EigenFaceRecognizer::create();
	}else if(arithmetic == 2){
		recognizer = cv::face::LBPHFaceRecognizer::create();
	}else{
		recognizer = cv::face::FisherFaceRecognizer::create();
	}

	std::cout << "Training ..." << std::endl;
	recognizer->train(images, labels);
	std::cout << "Train done ..." << std::endl;

	cv::Mat predict_image = cv::imread("test.jpg",cv::IMREAD_GRAYSCALE);

	std::cout << "Predicting ..." << std::endl;

	int predicted_label = -1;
	double confidence = 0.0;
	recognizer->predict(predict_image,predicted_label,confidence);

	if(predicted_label == 0){
		std::cout << "Matched person : Nick" << std::endl;
	}else if(predicted_label == 1){
		std::cout << "Matched person : Rick" << std::endl;
	}

	std::cout << "Confidence : "  << confidence << std::endl;

	return 0;
}
```



















# 开启GPU加速

```python
import cv2
import numpy as np

# 读取用于训练模型的数据并将其转为灰度图像
images = [cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE),cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)]

# 设置训练标签，几张人脸就几个标签，相当于起id
labels = [0, 1]

arithmetic = input(
'''
Chose Arithmetic realize :
1.Eigenfaces算法模型
'''
)

if arithmetic == "1":
    # 创建 Eigenfaces 算法模型
    recognizer = cv2.face.EigenFaceRecognizer_create()

    # 检查是否支持 GPU
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # 如果支持 GPU，则使用 GPU 进行训练
        print("Train with GPU...")
        recognizer = cv2.cuda_EigenFaceRecognizer.create()
        recognizer.train(images, np.array(labels))
    else:
        # 如果不支持 GPU，则使用 CPU 进行训练
        print("Train with CPU...")
        recognizer.train(images, np.array(labels))

else:
    print("Unsupported algorithm.")

# 加载一张待识别的人脸图像，并将其转化为灰度图像
predict_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# 对待识别图像进行人脸识别，并返回预测结果(label)和置信度(confidence)
label, confidence = recognizer.predict(predict_image)

if label == 0:
    print("匹配的人脸为尼根")
elif label == 1:
    print("匹配的人脸为瑞克")
print("confidence=", confidence)
```

