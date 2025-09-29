import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from dotenv import load_dotenv
import time

# .env 파일에서 환경 변수 로드
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
CRASH_DISTANCE_THRESHOLD = int(os.getenv("CRASH_DISTANCE_THRESHOLD", 100))

# 이메일 발송 함수
def send_alert_email(image_path):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
        print("이메일 설정이 누락되어 알림을 보낼 수 없습니다.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "!!긴급!! 교통사고 감지 알림"

        body = "실시간 영상에서 교통사고로 의심되는 상황이 감지되었습니다."
        msg.attach(MIMEText(body, 'plain'))

        with open(image_path, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(image_path))
        msg.attach(image)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
        server.quit()
        print(f"성공적으로 알림 이메일을 {RECEIVER_EMAIL} 주소로 발송했습니다.")
    except Exception as e:
        print(f"이메일 발송에 실패했습니다: {e}")

# 두 개의 바운딩 박스가 겹치는지 확인하는 함수
def check_intersection(box1, box2):
    # box: [x1, y1, x2, y2]
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def main():
    # YOLOv8 모델 로드 (사전 훈련된 모델 사용)
    model = YOLO('yolov8n.pt')

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 이메일 발송 쿨다운 (중복 발송 방지)
    last_alert_time = 0
    alert_cooldown = 60  # 초 단위

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8로 객체 탐지 수행
        results = model(frame)

        # 차량 클래스 ID (COCO 데이터셋 기준: 2=car, 3=motorcycle, 5=bus, 7=truck)
        vehicle_class_ids = [2, 3, 5, 7]
        
        detected_vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in vehicle_class_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_vehicles.append([x1, y1, x2, y2])

        crashed_indices = set()
        # 차량 쌍을 비교하여 충돌 감지
        if len(detected_vehicles) > 1:
            for i in range(len(detected_vehicles)):
                for j in range(i + 1, len(detected_vehicles)):
                    box1 = detected_vehicles[i]
                    box2 = detected_vehicles[j]

                    # 1. 바운딩 박스 겹침 확인
                    if check_intersection(box1, box2):
                        # 2. 중심점 간의 거리 확인
                        center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
                        center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
                        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

                        if distance < CRASH_DISTANCE_THRESHOLD:
                            crashed_indices.add(i)
                            crashed_indices.add(j)

        # 결과 프레임에 그리기
        for i, box in enumerate(detected_vehicles):
            x1, y1, x2, y2 = box
            if i in crashed_indices:
                # 충돌 감지 시 빨간색 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "CRASH", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # 일반 차량은 초록색 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 충돌이 감지되었고 쿨다운 시간이 지났으면 알림 발송
        if crashed_indices and (time.time() - last_alert_time > alert_cooldown):
            print("사고 감지! 알림을 준비합니다...")
            
            # 사고 장면 이미지 저장
            crash_image_path = "crash_detected.jpg"
            cv2.imwrite(crash_image_path, frame)
            
            # 이메일 발송
            send_alert_email(crash_image_path)
            
            # 마지막 알림 시간 갱신
            last_alert_time = time.time()

        # 화면에 영상 출력
        cv2.imshow('Real-time Accident Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()