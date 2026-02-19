# detect_webcam.py
from ultralytics import YOLO
import cv2

# 1) 모델 로드
model = YOLO("yolov8n.pt")  # 처음에만 모델 다운, 이후에는 캐시에서 사용

# 2) 웹캠 열기 (0 = 기본 캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 3) 객체 인식 (conf는 최소 신뢰도, 0.5 = 50%)
    results = model(frame, conf=0.5)

    # 4) 결과 박스 그린 프레임 얻기
    annotated_frame = results[0].plot()

    # 5) 화면에 보여주기
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # 6) q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
