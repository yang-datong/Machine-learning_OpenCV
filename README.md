# Machine-learning_OpenCV
机器学习-使用OpenCV框架

## 背景

需要科学上网。数据集下载链接:

```bash
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
```

下载项目后需要本地没有模型，运行文件会自定判断当前本地是否有模型

`haarcascade_frontalface_default.xml` 是OpenCV提供的人脸检测模型，用于从图片、摄像头中挑出人脸，后续的识别步骤需要根据不同的算法来进行训练模型，最后根据训练模型来识别人脸信息

## 使用Python
```bash
pip3 install opencv-contrib-python==4.7.0.72 numpy
python3 facedecetion.py
```

默认是调用电脑摄像头进行人脸识别，默认为LBPH训练模型，改模型需要改源代码


## 使用C++
```bash
cmake . && make && ./main
```

源代码为`main.cpp`和`main.hpp`,编译完成后执行`./main -h`即可看到使用帮助，默认模型为LBPH训练模型，修改模型可以直接传入参数命令
