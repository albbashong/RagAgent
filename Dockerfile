FROM python:3.12-slim

WORKDIR /app

# 필수 패키지 설치 + Node.js 20
RUN apt-get update && apt-get install -y \
    bash \
    curl gnupg ca-certificates build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python venv 생성 및 FastAPI 설치
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir fastapi uvicorn[standard]

# Node.js 의존성 설치
COPY package*.json ./
RUN npm install
COPY . .

# 포트 오픈
EXPOSE 9012 9013

# 엔트리포인트
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
