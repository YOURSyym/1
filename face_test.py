import cv2
from face_train import Model
import face_dataset
if __name__ == '__main__':
    # 加载模型
    model = Model()
    model.load_model(file_path='./data/me.face.model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    camera = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "C:\\Users\\ASUS\\Desktop\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"

    # 循环检测识别人脸
    while True:
        ret, frame = camera.read()  # 读取一帧视频

        # 图像灰化，降低计算复杂度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # 截取脸部图像提交给模型识别这是谁
                image = frame[y: y + h, x: x + w]
                image=face_dataset.resize_image(image)
                faceID = model.face_predict(image)

                # 如果是“我”
                if faceID == 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)

                    # 文字提示是谁
                    cv2.putText(frame, 'others',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)

                    # 文字提示是谁
                    cv2.putText(frame, 'prisoner',
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)

        cv2.imshow("camera", frame)

        # 等待1毫秒看是否有按键输入
        k = cv2.waitKey(1)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    camera.release()
    cv2.destroyAllWindows()
