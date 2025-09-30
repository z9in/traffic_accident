from flask import Flask, render_template, Response, request, session, redirect, url_for, flash, g
import cv2 
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import time
import queue
import sqlite3
import json
import click
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import yt_dlp

# .env 파일에서 환경 변수 로드
load_dotenv()
CRASH_DISTANCE_THRESHOLD = int(os.getenv("CRASH_DISTANCE_THRESHOLD", 100))

app = Flask(__name__)
app.config.from_mapping(
    DATABASE=os.path.join(app.instance_path, 'accident.sqlite'),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_default_secret_key_for_development")

# instance 폴더가 없으면 생성
if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)

# 전역 변수 설정
model = YOLO('yolov8n.pt')
video_streams = {}  # {stream_id: {'cap': cap_obj, 'title': '...', 'url': '...'}}

# --- Database Functions ---
def get_db():
    """현재 요청에 대한 데이터베이스 연결을 가져옵니다."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """데이터베이스 연결을 닫습니다."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """데이터베이스 테이블을 초기화합니다."""
    db = get_db()
    with app.open_resource('schema.sql') as f: # 'schema.sql' 파일을 읽어 테이블을 생성합니다.
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
def init_db_command():
    """CLI: 데이터베이스를 초기화합니다."""
    init_db()
    click.echo('데이터베이스가 초기화되었습니다.')

@click.command('add-user')
@click.argument('username')
@click.argument('password')
def add_user_command(username, password):
    """CLI: 새로운 사용자를 추가합니다."""
    db = get_db()
    try:
        db.execute(
            "INSERT INTO user (username, password) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        db.commit()
        click.echo(f"사용자 '{username}'이(가) 추가되었습니다.")
    except db.IntegrityError:
        click.echo(f"오류: 사용자 '{username}'은(는) 이미 존재합니다.")

app.teardown_appcontext(close_db)
app.cli.add_command(init_db_command)
app.cli.add_command(add_user_command)

def check_intersection(box1, box2):
    """두 개의 바운딩 박스가 겹치는지 확인하는 함수"""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def generate_frames():
    """비디오 프레임을 생성하고 사고를 감지하는 제너레이터 함수"""
    # 이 함수는 더 이상 직접 사용되지 않습니다. generate_frames_for_stream으로 대체됩니다.
    pass

def generate_frames_for_stream(stream_id):
    """특정 스트림 ID에 대한 비디오 프레임을 생성하고 사고를 감지합니다."""
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
            alert_message = f"[{stream_info.get('title', 'Unknown Video')}] 교통사고 의심!"
            
            # SSE 스트림으로 직접 보낼 알림 데이터 생성
            alert_data = {"message": alert_message, "timestamp": timestamp}
            last_alert_time = time.time()
            app.alert_queue.put(alert_data) # 큐에 알림 추가
            print(f"[{timestamp}] 사고 감지! 웹페이지에 알림을 보냅니다.")

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 스트리밍을 위해 프레임 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    """메인 페이지를 렌더링합니다. 로그인된 사용자만 접근 가능합니다."""    
    global video_streams
    
    if g.user is None:
        return redirect(url_for('login'))

    # POST 요청 처리 로직은 그대로 유지
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    """로그인 페이지를 처리합니다."""
    if g.user:
        return redirect(url_for('index'))

    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute(
            'SELECT * FROM user WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = '등록되지 않은 사용자입니다.'
        elif not check_password_hash(user['password'], password):
            error = '잘못된 비밀번호입니다.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('index'))
        else:
            flash(error)

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """로그아웃을 처리합니다."""
    session.clear()
    flash('로그아웃되었습니다.')
    return redirect(url_for('login'))


@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    """비디오 스트리밍 경로"""
    if g.user is None:
        return "Unauthorized", 401
    return Response(generate_frames_for_stream(stream_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    """실시간 알림을 위한 Server-Sent Events 경로"""
    if not hasattr(app, 'alert_queue'):
        app.alert_queue = queue.Queue()

    def event_stream():
        while True:
            try:
                # 큐에서 새로운 알림을 기다림 (블로킹)
                alert_data = app.alert_queue.get(timeout=60) 
                data = json.dumps(alert_data)
                yield f"data: {data}\n\n"
            except queue.Empty:
                # 타임아웃 동안 메시지가 없으면 연결 유지를 위해 주석 메시지 전송
                yield ": keep-alive\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.before_request
def load_logged_in_user():
    """요청이 처리되기 전에 로그인된 사용자를 확인합니다."""
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()

@app.route('/about')
def about():
    """소개 페이지를 보여줍니다."""
    return "<h1>About</h1><p>이 시스템은 YOLOv8을 활용한 실시간 교통사고 감지 시스템입니다.</p>"

if __name__ == '__main__':
    # host='0.0.0.0'으로 설정하면 동일 네트워크의 다른 기기에서도 접속 가능
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)