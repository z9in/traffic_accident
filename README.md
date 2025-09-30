# 실시간 교통사고 감지 시스템 (Real-time Traffic Accident Detection System)

YOLOv8 모델을 활용하여 YouTube와 같은 실시간 비디오 스트림에서 교통사고를 감지하고 웹을 통해 알림을 제공하는 시스템입니다.


## 🌟 주요 기능 (Key Features)

- **실시간 영상 분석**: 다수의 YouTube 영상 URL을 입력받아 동시에 실시간으로 분석합니다.
- **YOLOv8 기반 객체 탐지**: 최신 YOLOv8 모델을 사용하여 영상 속 차량을 탐지하고, 차량 간의 충돌을 감지합니다.
- **웹 기반 인터페이스**: Flask 기반의 웹 서버를 통해 사용자가 쉽게 접근하고 모니터링할 수 있습니다.
- **즉각적인 알림**: 사고 의심 상황이 감지되면 웹 페이지에 목록과 팝업 모달 형태로 즉시 알림을 표시합니다.
- **사용자 인증**: 안전한 접근을 위해 로그인/로그아웃 기능을 제공하며, 사용자 정보는 데이터베이스에 안전하게 저장됩니다.
- **성능 최적화**: 프레임 건너뛰기, 해상도 조절 등의 기법을 적용하여 저사양 환경에서도 원활하게 동작하도록 최적화되었습니다.

## 🛠️ 기술 스택 (Tech Stack)

- **Backend**: Python, Flask
- **AI/ML**: PyTorch, Ultralytics YOLOv8
- **Video Processing**: OpenCV, yt-dlp
- **Database**: SQLite
- **Frontend**: HTML, CSS, JavaScript (Server-Sent Events)

## ⚙️ 설치 및 실행 방법 (Installation & Usage)

### 1. 저장소 복제 (Clone Repository)
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. 가상 환경 생성 및 활성화 (Optional, but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. 필요 라이브러리 설치 (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (Environment Variables)
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 참고하여 키를 설정합니다.

```env
# 사고 감지 민감도 (값이 작을수록 민감)
CRASH_DISTANCE_THRESHOLD=100

# Flask 세션 암호화를 위한 시크릿 키
FLASK_SECRET_KEY=a_very_secret_key_for_production
```

### 5. 데이터베이스 초기화 (Initialize Database)
최초 실행 시 한 번만 필요합니다. 터미널에 다음 명령어를 입력하여 사용자 정보를 저장할 데이터베이스 테이블을 생성합니다.

```bash
python -m flask init-db
```

### 6. 초기 사용자 추가 (Add Initial User)
로그인에 사용할 계정을 생성합니다.

```bash
# 예시: 사용자 이름 'admin', 비밀번호 '1234'
python -m flask add-user admin 1234
```

### 7. 서버 실행 (Run Server)
```bash
python app.py
```

서버가 실행되면 웹 브라우저에서 `http://127.0.0.1:5000` 또는 `http://localhost:5000`으로 접속하여 로그인 후 시스템을 사용할 수 있습니다.

## 📄 라이선스 (License)

이 프로젝트는 **AGPL-3.0 라이선스**를 따릅니다.

이 프로젝트는 Ultralytics의 YOLOv8을 사용하고 있으며, YOLOv8은 AGPL-3.0 라이선스가 적용되어 있습니다. 따라서 AGPL-3.0 라이선스의 요구사항에 따라 이 프로젝트 또한 동일한 라이선스 하에 소스 코드를 공개합니다.

자세한 내용은 `LICENSE` 파일을 참고하십시오.

---

### 원본 YOLOv8 저작권 정보 (Original YOLOv8 Copyright)

*   **Copyright**: © 2023 Ultralytics. All rights reserved.
*   **License**: AGPL-3.0

*   **Repository**: https://github.com/ultralytics/ultralytics
