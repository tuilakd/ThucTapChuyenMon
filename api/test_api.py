# -*- coding: utf-8 -*-
"""
api/test_api.py — Bước 5: Chạy thử và kiểm tra API (PDF Mục 3.8).

Kịch bản kiểm thử toàn diện:
- Kiểm tra các ca hợp lệ (sinh viên chuyên hướng AI, Bảo mật, Web).
- Kiểm tra các ca không hợp lệ: điểm âm, điểm > 10, thiếu trường môn học, sai kiểu dữ liệu.
- So sánh HTTP status code nhận được với mã mong đợi.
- Đo thời gian phản hồi (ms) cho từng ca.
- Tự động phát hiện nếu server đang chạy trên cổng 8000 (dùng requests), hoặc fallback dùng TestClient.
"""
import os
import sys
import time
import json
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "api" else CURRENT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Các bộ dữ liệu kiểm thử
TEST_CASES = [
    {
        "name": "Health check",
        "method": "GET",
        "endpoint": "/health",
        "payload": None,
        "expected_status": 200,
    },
    {
        "name": "Lấy Model Info (Metadata)",
        "method": "GET",
        "endpoint": "/model-info",
        "payload": None,
        "expected_status": 200,
    },
    {
        "name": "SV thế mạnh Dữ liệu & AI",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.5, "linux": 7.0, "db": 9.0, "graphics": 6.0,
            "network": 7.0, "analysis": 8.0, "image": 8.5, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": 9.5,
            "skills": {"self_learning": "excellent", "english": "strong"},
            "interests": ["Dữ liệu & AI"],
            "student_name": "Nguyễn Văn A (Mạnh AI)",
        },
        "expected_status": 200,
        "expected_group": "Dữ liệu & AI",
    },
    {
        "name": "SV thế mạnh An toàn & Bảo mật",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 9.5, "se": 7.0, "linux": 9.0, "db": 7.0, "graphics": 6.0,
            "network": 9.0, "analysis": 7.5, "image": 6.5, "java": 7.0,
            "csharp": 7.0, "web": 6.5, "ai": 7.0,
            "skills": {"problem_solving": "excellent", "english": "strong"},
            "interests": ["An toàn thông tin"],
            "student_name": "Trần Thị B (Mạnh Bảo Mật)",
        },
        "expected_status": 200,
        "expected_group": "An toàn thông tin",
    },
    {
        "name": "SV thế mạnh Phát triển Web",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 6.5, "se": 8.0, "linux": 7.0, "db": 7.5, "graphics": 8.0,
            "network": 7.0, "analysis": 7.5, "image": 7.0, "java": 8.0,
            "csharp": 8.0, "web": 9.8, "ai": 6.5,
            "skills": {"teamwork": "strong", "communication": "strong"},
            "interests": ["Phát triển Web"],
            "student_name": "Lê Văn C (Mạnh Web)",
        },
        "expected_status": 200,
        "expected_group": "Phát triển Web",
    },
    {
        "name": "Điểm âm (ai = -2.0)",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": -2.0,
        },
        "expected_status": 422,
    },
    {
        "name": "Điểm vượt trần (db = 15.0)",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.0, "linux": 7.0, "db": 15.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": 8.0,
        },
        "expected_status": 422,
    },
    {
        "name": "Thiếu trường môn học (chỉ gửi 10 môn)",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0, "csharp": 7.0,
        },
        "expected_status": 422,
    },
    {
        "name": "Sai kiểu dữ liệu (điểm là chuỗi chữ 'abc')",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": "abc", "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": 8.0,
        },
        "expected_status": 422,
    },
    {
        "name": "Sở thích không có trong danh mục",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": 8.0,
            "interests": ["Nghề không có thật"],
        },
        "expected_status": 422,
    },
    {
        "name": "Kỹ năng mềm không có trong danh mục",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": 8.0,
            "skills": {"khong_ton_tai": "strong"},
        },
        "expected_status": 422,
    },
    {
        "name": "Cấp độ kỹ năng không hợp lệ",
        "method": "POST",
        "endpoint": "/decision",
        "payload": {
            "sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0,
            "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0,
            "csharp": 7.0, "web": 7.0, "ai": 8.0,
            "skills": {"teamwork": "sieu_gioi"},
        },
        "expected_status": 422,
    },
    {"name": "Biên dưới (toàn 0.0)", "method": "POST", "endpoint": "/predict",
     "payload": {"sec": 0.0, "se": 0.0, "linux": 0.0, "db": 0.0, "graphics": 0.0, "network": 0.0, "analysis": 0.0, "image": 0.0, "java": 0.0, "csharp": 0.0, "web": 0.0, "ai": 0.0}, "expected_status": 200},
    {"name": "Biên trên (toàn 10.0)", "method": "POST", "endpoint": "/predict",
     "payload": {"sec": 10.0, "se": 10.0, "linux": 10.0, "db": 10.0, "graphics": 10.0, "network": 10.0, "analysis": 10.0, "image": 10.0, "java": 10.0, "csharp": 10.0, "web": 10.0, "ai": 10.0}, "expected_status": 200},
    {"name": "Giá trị boolean (ai = true)", "method": "POST", "endpoint": "/decision",
     "payload": {"sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0, "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0, "csharp": 7.0, "web": 7.0, "ai": True}, "expected_status": 422},
    {"name": "Họ tên quá dài (>100 ký tự)", "method": "POST", "endpoint": "/decision",
     "payload": {"sec": 7.0, "se": 7.0, "linux": 7.0, "db": 7.0, "graphics": 7.0, "network": 7.0, "analysis": 7.0, "image": 7.0, "java": 7.0, "csharp": 7.0, "web": 7.0, "ai": 7.0, "student_name": "A" * 500}, "expected_status": 422},
]


def run_tests():
    # Kiểm tra xem server có đang chạy trực tiếp trên port hay không
    server_online = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        if r.status_code == 200:
            server_online = True
    except Exception:
        server_online = False

    client = None
    if not server_online:
        print(f"[*] Máy chủ API chưa chạy tại {API_URL}. Sử dụng FastAPI TestClient để chạy kiểm thử nội bộ...")
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
    else:
        print(f"[*] Kết nối thành công đến API đang chạy tại {API_URL}...")

    print("=" * 78)
    print("BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG BỘ REST API (Bước 5 - PDF Mục 3.8)")
    print("=" * 78)

    all_passed = True

    for tc in TEST_CASES:
        name = tc["name"]
        method = tc["method"]
        endpoint = tc["endpoint"]
        payload = tc["payload"]
        exp_status = tc["expected_status"]

        t0 = time.time()
        if server_online:
            url = f"{API_URL}{endpoint}"
            if method == "GET":
                res = requests.get(url, timeout=5)
            else:
                res = requests.post(url, json=payload, timeout=10)
            status_code = res.status_code
            try:
                body = res.json()
            except Exception:
                body = {}
        else:
            if method == "GET":
                res = client.get(endpoint)
            else:
                res = client.post(endpoint, json=payload)
            status_code = res.status_code
            try:
                body = res.json()
            except Exception:
                body = {}

        latency_ms = round((time.time() - t0) * 1000)

        is_ok = (status_code == exp_status)
        if not is_ok:
            all_passed = False

        status_tag = "[OK ]" if is_ok else "[FAIL]"
        print(f"{status_tag} {name:<40} HTTP {status_code} (mong đợi {exp_status}) - {latency_ms:>3} ms")

        # In thêm thông tin chi tiết nếu là ca hợp lệ
        if status_code == 200 and "predicted_group" in body:
            pred_group = body.get("predicted_group")
            top1_c = body.get("top_careers", [{}])[0].get("career_name", "N/A")
            top1_m = body.get("top_careers", [{}])[0].get("match_percentage", 0)
            print(f"      -> Dự đoán: {pred_group} | Nghề #1: {top1_c} ({top1_m}%)")

        if not is_ok:
            print(f"      -> Lỗi: Phản hồi không khớp kỳ vọng. Chi tiết: {body}")

    print("=" * 78)
    if all_passed:
        print("[✓] TẤT CẢ CÁC CA KIỂM THỬ ĐỀU ĐẠT CHUẨN!")
    else:
        print("[X] CÓ CA KIỂM THỬ THẤT BẠI. Vui lòng kiểm tra lại log chi tiết ở trên.")
    print("=" * 78)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
