
def detect_face():
    # 调用分类器
    face_cascade = cv2.CascadeClassifier('C:\\Users\\ASUS\\Desktop\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    # while循环，一帧一帧将读取视频流数据
    while True:
        ret, frame = camera.read()
        # 将图片灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        # x,y为矩形的对角线顶点的坐标，w为宽度，h为高度
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
