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
