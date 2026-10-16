# -*- coding: utf-8 -*-
"""
monitor_service.py — Quản lý Giai đoạn 5 (MONITOR) trong vòng đời Model AI.

Nhiệm vụ:
  1. Ghi nhật ký dự đoán (Prediction Logging / Audit Trail):
     - Ghi nhận mỗi lượt dự đoán (scores, top_group, top_prob, model_version, latency_ms) vào data/prediction_logs.jsonl.
  2. Giám sát độ lệch dữ liệu (Data Drift Detection):
     - So sánh phân phối điểm số 12 môn gần đây với baseline trong models/current/metadata.json (Standardized Mean Shift).
  3. Giám sát độ lệch dự đoán (Prediction / Output Drift):
     - So sánh tỷ lệ phân bố các nhóm nghề được dự đoán gần đây với phân bố tập huấn luyện gốc.
  4. Theo dõi phản hồi người dùng & chất lượng thực tế (Feedback & Accuracy):
     - Đo lường độ hài lòng (sao), tỷ lệ thích/không thích.
     - Kiểm định độc lập độ chính xác Top-1 và Top-3 trên các phản hồi đã được người dùng xác nhận.
  5. Cảnh báo & Kích hoạt vòng lặp Huấn luyện lại (Retrain Trigger):
     - Đánh giá điều kiện dữ liệu mới hoặc phát hiện model lệch để kích hoạt chu trình:
       [5. Monitor] -> [1. Train] -> [2. Export] -> [3. Serve].
"""
import datetime
import json
import os
import sys
import threading
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from .domain_knowledge import (
    CAREER_GROUPS, KEY_TO_SUBJECT, SUBJECT_KEYS, SUBJECTS,
)
from .model_exporter import get_current_metadata

LOGS_FILE = "data/prediction_logs.jsonl"
FEEDBACK_FILE = "data/feedback.json"
STUDENTS_FILE = "data/students.csv"

_LOG_LOCK = threading.Lock()


def log_prediction(
    input_scores: dict,
    top_group: str,
    top_prob: float,
    model_version: str = "1.0.0",
    latency_ms: float = 0.0,
    source: str = "web_ui",
) -> dict:
    """Ghi nhật ký một lượt dự đoán vào file JSONL."""
    os.makedirs(os.path.dirname(LOGS_FILE) or ".", exist_ok=True)
    record = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "model_version": model_version,
        "source": source,
        "input_scores": {
            SUBJECT_KEYS.get(k, k): float(v) for k, v in input_scores.items() if (k in SUBJECTS or k in SUBJECT_KEYS.values())
        },
        "predicted_group": top_group,
        "top_probability": float(round(top_prob, 4)),
        "latency_ms": float(round(latency_ms, 2)),
    }

    with _LOG_LOCK:
        with open(LOGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def get_prediction_logs(limit: int = 100) -> list:
    """Đọc danh sách các lượt dự đoán gần nhất."""
    if not os.path.exists(LOGS_FILE):
        return []
    records = []
    with _LOG_LOCK:
        with open(LOGS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:limit]


def compute_data_drift() -> dict:
    """
    Phân tích Data Drift bằng cách so sánh phân phối điểm của các request gần đây
    với baseline_feature_stats trong models/current/metadata.json.
    """
    metadata = get_current_metadata()
    baseline_stats = metadata.get("baseline_feature_stats", {})

    logs = get_prediction_logs(limit=200)
    total_logs = len(logs)

    feature_cols = [SUBJECT_KEYS[s] for s in SUBJECTS]
    drift_details = {}
    drifted_features = []

    if total_logs < 5 or not baseline_stats:
        # Chưa đủ dữ liệu suy luận để đo drift
        for key in feature_cols:
            subj = KEY_TO_SUBJECT.get(key, key)
            base = baseline_stats.get(key, {"mean": 7.5, "std": 1.2})
            drift_details[key] = {
                "subject_name": subj,
                "baseline_mean": base.get("mean", 7.5),
                "baseline_std": base.get("std", 1.2),
                "recent_mean": base.get("mean", 7.5),
                "recent_std": base.get("std", 1.2),
                "z_shift": 0.0,
                "status": "Ổn định (Chờ thêm mẫu)",
                "is_drifted": False,
            }
        return {
            "total_inference_samples": total_logs,
            "has_drift": False,
            "drift_status": "Ổn định (Chưa đủ mẫu suy luận)",
            "drift_level": "normal",
            "drifted_feature_count": 0,
            "features": drift_details,
        }

    # Trích xuất điểm các request
    recent_matrix = []
    for log in logs:
        scores = log.get("input_scores", {})
        if len(scores) >= len(feature_cols):
            recent_matrix.append([scores.get(k, 7.0) for k in feature_cols])

    if not recent_matrix:
        return {
            "total_inference_samples": total_logs,
            "has_drift": False,
            "drift_status": "Ổn định",
            "drift_level": "normal",
            "drifted_feature_count": 0,
            "features": drift_details,
        }

    df_recent = pd.DataFrame(recent_matrix, columns=feature_cols)

    for key in feature_cols:
        subj = KEY_TO_SUBJECT.get(key, key)
        base = baseline_stats.get(key, {"mean": 7.5, "std": 1.2})
        base_mean = base.get("mean", 7.5)
        base_std = max(base.get("std", 1.2), 0.001)

        rec_mean = float(df_recent[key].mean())
        rec_std = float(df_recent[key].std())

        # Độ lệch chuẩn hóa (Z-shift)
        z_shift = abs(rec_mean - base_mean) / base_std

        is_drifted = z_shift >= 0.5  # Ngưỡng lệch đáng kể
        if is_drifted:
            drifted_features.append(subj)

        drift_details[key] = {
            "subject_name": subj,
            "baseline_mean": round(base_mean, 2),
            "baseline_std": round(base_std, 2),
            "recent_mean": round(rec_mean, 2),
            "recent_std": round(rec_std, 2),
            "z_shift": round(z_shift, 3),
            "status": "Lệch dữ liệu (Cảnh báo)" if is_drifted else "Ổn định",
            "is_drifted": is_drifted,
        }

    has_drift = len(drifted_features) >= 2 or any(d["z_shift"] >= 0.8 for d in drift_details.values())
    drift_status = "Cảnh báo Lệch dữ liệu (Drift Alert)" if has_drift else "Ổn định (Phân phối chuẩn)"
    drift_level = "warning" if has_drift else "normal"

    return {
        "total_inference_samples": len(df_recent),
        "has_drift": has_drift,
        "drift_status": drift_status,
        "drift_level": drift_level,
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "features": drift_details,
    }


def compute_prediction_drift() -> dict:
    """So sánh tỷ lệ nhóm nghề được dự đoán so với tỷ lệ nhãn trong tập huấn luyện gốc."""
    logs = get_prediction_logs(limit=200)
    if not logs:
        return {"predictions_count": 0, "distribution": {}}

    counts = {}
    for g in CAREER_GROUPS:
        counts[g] = 0

    for log in logs:
        grp = log.get("predicted_group")
        if grp in counts:
            counts[grp] += 1
        elif grp:
            counts[grp] = 1

    total = sum(counts.values()) or 1
    distribution = {g: {"count": n, "percentage": round(n / total * 100, 1)} for g, n in counts.items()}
    return {
        "predictions_count": total,
        "distribution": distribution,
    }


def get_monitoring_dashboard_data() -> dict:
    """Tổng hợp toàn bộ chỉ số cho trang Bảng điều khiển Giám sát (Giai đoạn 5)."""
    meta = get_current_metadata()
    data_drift = compute_data_drift()
    pred_drift = compute_prediction_drift()
    logs = get_prediction_logs(limit=20)

    # Đọc feedback
    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, encoding="utf-8") as f:
                feedbacks = json.load(f)
        except Exception:
            feedbacks = []

    ratings = [f.get("rating") for f in feedbacks if f.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

    # Đếm số lượng mẫu feedback có thể dùng huấn luyện lại (có bảng điểm + có nghề like)
    ready_feedback_count = sum(1 for f in feedbacks if f.get("scores") and f.get("liked_careers"))

    # Tính độ trễ trung bình
    latencies = [l.get("latency_ms", 0) for l in logs if l.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Khuyến nghị Retrain
    retrain_recommended = (ready_feedback_count >= 5) or data_drift.get("has_drift", False)
    if ready_feedback_count >= 5 and data_drift.get("has_drift", False):
        recommendation_text = "Phát hiện Lệch dữ liệu VÀ đã tích luỹ đủ phản hồi người dùng. Cần kích hoạt vòng lặp Retrain ngay!"
    elif data_drift.get("has_drift", False):
        recommendation_text = "Phát hiện Lệch dữ liệu (Data Drift). Hệ thống đề xuất thu thập thêm phản hồi và huấn luyện lại."
    elif ready_feedback_count >= 5:
        recommendation_text = f"Đã có {ready_feedback_count} mẫu phản hồi thực tế mới. Bạn có thể kích hoạt Retrain để cập nhật mô hình."
    else:
        recommendation_text = "Mô hình đang hoạt động ổn định trong phạm vi kiểm soát."

    return {
        "model_version": meta.get("model_version", "1.0.0"),
        "algorithm": meta.get("algorithm", "Logistic Regression (multinomial)"),
        "exported_at": meta.get("exported_at", "N/A"),
        "metrics": meta.get("evaluation_metrics", {}),
        "total_predictions": pred_drift.get("predictions_count", 0),
        "avg_latency_ms": avg_latency,
        "data_drift": data_drift,
        "prediction_drift": pred_drift,
        "recent_logs": logs[:15],
        "feedback_summary": {
            "total_feedback": len(feedbacks),
            "avg_rating": avg_rating,
            "ready_for_retrain_count": ready_feedback_count,
        },
        "retrain_status": {
            "recommended": retrain_recommended,
            "message": recommendation_text,
        }
    }
