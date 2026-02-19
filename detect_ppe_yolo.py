# detect_ppe_yolo.py

from ultralytics import YOLO
import cv2
import json

# 1) YOLOv8 PPE 모델 경로 (GitHub에서 받은 best.pt)
MODEL_PATH = "best.pt"  # 파일 위치가 다르면 경로 수정

# 2) 관심 있는 클래스만 (이 데이터셋 기준 10개 클래스 정도 존재)
#   Roboflow Construction Site Safety 데이터셋 클래스 예시:
#   ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
#    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
USE_CLASSES = {
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Safety Cone",
    "machinery",
    "vehicle",
}

# 3) 분석할 이미지 경로
IMAGE_PATH = "site.jpg"  # 공사현장 사진 파일 이름으로 바꾸기


def main():
    # 이미지 불러오기
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    # 모델 로드
    print("[INFO] 모델 로딩 중...")
    model = YOLO(MODEL_PATH)

    # 클래스 이름 확인 (처음에 한 번 프린트해서 실제 이름 체크)
    print("[INFO] 모델 클래스 목록:", model.names)

    # 추론
    print("[INFO] 추론 실행 중...")
    results = model(img)

    detections = []  # 구조화된 결과 저장용 (나중에 예측 모델/DB에 쓰기 좋게)

    r = results[0]
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        # 쓸 클래스만 필터링 (나머지는 버리기)
        if cls_name not in USE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )

        # 시각화 (박스 + 라벨)
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )
        label = f"{cls_name} {conf * 100:.1f}%"
        cv2.putText(
            img,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # 결과 이미지 보여주기
    cv2.imshow("PPE Detection (YOLOv8)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 콘솔에 JSON 형태로도 출력 (나중에 공사현장 위험 예측 로직에 쓰기 좋게)
    print("[INFO] detections(JSON):")
    print(json.dumps(detections, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
