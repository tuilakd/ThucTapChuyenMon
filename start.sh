#!/bin/bash
# Khởi động đồng thời FastAPI Backend và Streamlit Frontend.
# Render yêu cầu tiến trình public lắng nghe trên $PORT.
PORT="${PORT:-8501}"
API_PORT="${API_PORT:-8000}"
API_URL="${API_URL:-http://127.0.0.1:${API_PORT}}"

echo "Đang khởi động FastAPI Backend trên cổng ${API_PORT}..."
uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

cleanup() {
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Chờ FastAPI sẵn sàng
sleep 2

echo "Đang khởi động Streamlit Frontend trên cổng ${PORT}..."
API_URL="${API_URL}" streamlit run web/app.py --server.port "${PORT}" --server.address 0.0.0.0
