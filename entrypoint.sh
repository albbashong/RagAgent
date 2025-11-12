#!/bin/bash
set -e

service postgresql start

# Node.js 서버 실행 (9012번 포트)
node server.js &

# Uvicorn 서버 실행 (9013번 포트)
uvicorn bridge_server:app --host 0.0.0.0 --port 9013 --reload



# foreground 유지
wait -n