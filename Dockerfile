# --- Stage 1: Builder ---
# 빌드에 필요한 모든 도구와 라이브러리를 포함하는 빌더 이미지
FROM python:3.11-slim as builder

# 시스템 패키지 업데이트 및 빌드에 필요한 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 가상 환경 생성
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# requirements.txt 복사 및 의존성 설치
WORKDIR /app
COPY requirements.txt .
# PyTorch와 OpenCV를 최적화된 버전으로 설치하여 용량 감소
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final ---
# 실제 애플리케이션을 실행할 가볍고 깨끗한 이미지
FROM python:3.11-slim

# 시스템 패키지 업데이트 및 OpenCV 실행에 필요한 라이브러리만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 빌더 이미지에서 생성된 가상 환경 복사
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 애플리케이션 코드 복사
WORKDIR /app
COPY . .

# 컨테이너 실행 시 Flask 애플리케이션 구동
EXPOSE 5000
CMD ["python", "app.py"]