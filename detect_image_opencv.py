import cv2
import numpy as np

# 1) 클래스 이름 (모델이 학습된 20개 카테고리)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# 2) 모델 파일 경로
PROTOTXT = "MobileNetSSD_deploy.prototxt"       # 여기에 다운받은 prototxt 이름
MODEL = "MobileNetSSD_deploy.caffemodel"        # 여기에 다운받은 caffemodel 이름

# 3) 네트워크 로드
print("[INFO] 모델 로딩 중...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# 4) 이미지 불러오기
image_path = "test.jpg"   # 같은 폴더에 test.jpg 넣어두기
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

(h, w) = image.shape[:2]

# 5) 전처리 (blob 생성)
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    0.007843,          # 스케일
    (300, 300),
    127.5
)

# 6) 네트워크에 입력 후 forward
net.setInput(blob)
detections = net.forward()

# 7) 결과 반복하면서 박스 그리기
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # 신뢰도 threshold
    if confidence > 0.4:
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx] if idx < len(CLASSES) else "unknown"

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = f"{label}: {confidence * 100:.1f}%"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 8) 결과 보여주기
cv2.imshow("Object Detection (OpenCV DNN)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
