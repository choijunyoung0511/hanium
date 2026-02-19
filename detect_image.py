# detect_image.py
from ultralytics import YOLO
import cv2

# 1) YOLOv8 기본 모델 불러오기 (처음 실행 시 자동 다운로드)
model = YOLO("yolov8n.pt")  # n = nano, 가장 가벼운 모델

# 2) 이미지 불러오기
image_path = "test.jpg"  # 같은 폴더에 test.jpg 넣어두기
img = cv2.imread(image_path)

# 3) 객체 인식 실행
results = model(img)

# 4) 결과 그리기 (bounding box + 라벨)
annotated_img = results[0].plot()

# 5) 화면에 보여주기
cv2.imshow("Object Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
