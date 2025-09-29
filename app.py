from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import time
import json
from datetime import datetime
import yt_dlp

# .env 파일에서 환경 변수 로드
load_dotenv()
CRASH_DISTANCE_THRESHOLD = int(os.getenv("CRASH_DISTANCE_THRESHOLD", 100))

app = Flask(__name__)

# 전역 변수 설정
model = YOLO('yolov8n.pt')
video_streams = {}  # {stream_id: {'cap': cap_obj, 'title': '...', 'url': '...'}}

latest_alert = {
    "message": None,
    "timestamp": None
}

def check_intersection(box1, box2):
    """두 개의 바운딩 박스가 겹치는지 확인하는 함수"""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def generate_frames():
    """비디오 프레임을 생성하고 사고를 감지하는 제너레이터 함수"""
    # 이 함수는 더 이상 직접 사용되지 않습니다. generate_frames_for_stream으로 대체됩니다.
    pass

def generate_frames_for_stream(stream_id):
    """특정 스트림 ID에 대한 비디오 프레임을 생성하고 사고를 감지합니다."""
    global latest_alert
    last_alert_time = 0
    alert_cooldown = 10  # 초 단위 (웹 알림은 더 짧게 설정 가능)

    stream_info = video_streams.get(stream_id)

    # --- 성능 측정용 변수 ---
    frame_count_for_fps = 0
    fps_start_time = time.time()
    display_fps = 0

    # --- 최적화 설정 ---
    TARGET_FPS = 5  # 1초에 5 프레임만 분석 (성능 향상)
    FRAME_SKIP_INTERVAL = int(stream_info.get('cap').get(cv2.CAP_PROP_FPS) / TARGET_FPS) if TARGET_FPS > 0 else 1

    if not stream_info or not stream_info['cap'].isOpened():
        return

    cap = stream_info['cap']
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if stream_info['url'] and 'googlevideo.com' in stream_info['url']: # 유튜브 스트림은 재시작
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # --- 프레임 건너뛰기 (성능 향상) ---
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_count % FRAME_SKIP_INTERVAL != 0:
            continue
        
        # --- 해상도 줄이기 (성능 향상) ---
        frame = cv2.resize(frame, (640, 480))

        # YOLOv8로 객체 탐지
        results = model(frame, verbose=False)

        vehicle_class_ids = [2, 3, 5, 7]
        detected_vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) in vehicle_class_ids:
                    detected_vehicles.append(list(map(int, box.xyxy[0])))

        crashed_indices = set()
        if len(detected_vehicles) > 1:
            for i in range(len(detected_vehicles)):
                for j in range(i + 1, len(detected_vehicles)):
                    box1, box2 = detected_vehicles[i], detected_vehicles[j]
                    if check_intersection(box1, box2):
                        center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
                        center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
                        distance = np.linalg.norm(np.array(center1) - np.array(center2))

                        if distance < CRASH_DISTANCE_THRESHOLD:
                            crashed_indices.update([i, j])

        # 프레임에 결과 그리기
        for i, box in enumerate(detected_vehicles):
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if i in crashed_indices else (0, 255, 0)
            label = "CRASH" if i in crashed_indices else "Vehicle"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # --- 성능(FPS) 계산 및 표시 ---
        frame_count_for_fps += 1
        # 10프레임마다 FPS를 갱신하여 표시
        if frame_count_for_fps >= 10:
            elapsed_time = time.time() - fps_start_time
            display_fps = frame_count_for_fps / elapsed_time
            # 카운터와 시간 초기화
            frame_count_for_fps = 0
            fps_start_time = time.time()
        
        cv2.putText(frame, f"Proc FPS: {display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 충돌 감지 및 알림 생성
        if crashed_indices and (time.time() - last_alert_time > alert_cooldown):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            latest_alert["message"] = f"[{stream_info.get('title', 'Unknown Video')}] 교통사고 의심!"
            latest_alert["timestamp"] = timestamp
            last_alert_time = time.time()
            print(f"[{timestamp}] 사고 감지! 웹페이지에 알림을 보냅니다.")

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 스트리밍을 위해 프레임 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    """메인 페이지를 렌더링합니다."""
    global video_streams
    error_message = None
    youtube_urls_text = ""

    if request.method == 'POST':
        # 기존 스트림 모두 정리
        for stream_id, info in video_streams.items():
            if info['cap']:
                info['cap'].release()
        video_streams.clear()

        youtube_urls_text = request.form.get('youtube_urls', '')
        urls = [url.strip() for url in youtube_urls_text.splitlines() if url.strip()]

        if not urls:
            error_message = "분석할 URL을 입력해주세요."
        else:
            for i, url in enumerate(urls):
                try:
                    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        stream_url = info['url']
                    
                    cap = cv2.VideoCapture(stream_url)
                    if cap.isOpened():
                        video_streams[i] = {
                            'cap': cap,
                            'title': info.get('title', 'N/A'),
                            'url': stream_url,
                            'fps': cap.get(cv2.CAP_PROP_FPS)
                        }
                        print(f"스트림 {i} 분석 시작: {info.get('title', 'N/A')}")
                    else:
                        raise Exception("cv2.VideoCapture에서 스트림을 열 수 없습니다.")
                except Exception as e:
                    error_message = f"URL 로드 실패: {url[:30]}... ({e})"
                    print(error_message)
                    # 하나라도 실패하면 모든 스트림을 중단하고 오류 메시지 표시
                    for stream_id, info in video_streams.items():
                        if info['cap']: info['cap'].release()
                    video_streams.clear()
                    break
    
    return render_template('index.html', streams=video_streams, youtube_urls_text=youtube_urls_text, error_message=error_message)

@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    """비디오 스트리밍 경로"""
    return Response(generate_frames_for_stream(stream_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    """실시간 알림을 위한 Server-Sent Events 경로"""
    def event_stream():
        last_sent_timestamp = None
        while True:
            # 새로운 알림이 있고, 이전에 보낸 알림과 다를 경우에만 전송
            if latest_alert["timestamp"] and latest_alert["timestamp"] != last_sent_timestamp:
                data = json.dumps(latest_alert)
                yield f"data: {data}\n\n"
                last_sent_timestamp = latest_alert["timestamp"]
            time.sleep(1) # 1초마다 새로운 알림 확인
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/about')
def about():
    """소개 페이지를 보여줍니다."""
    return "<h1>About</h1><p>이 시스템은 YOLOv8을 활용한 실시간 교통사고 감지 시스템입니다.</p>"

if __name__ == '__main__':
    # host='0.0.0.0'으로 설정하면 동일 네트워크의 다른 기기에서도 접속 가능
    app.run(debug=True, host='0.0.0.0', port=5000)