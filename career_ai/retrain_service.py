# -*- coding: utf-8 -*-
"""
retrain_service.py — Dịch vụ tự động huấn luyện lại mô hình dựa trên phản hồi tường minh (Explicit Feedback).

CƠ CHẾ:
  1. Đọc data/feedback.json, lọc các bản ghi có đánh giá "Phù hợp" (liked_careers)
     kèm theo bảng điểm 12 môn của sinh viên.
  2. Ánh xạ các nghề được đánh giá "Phù hợp" về nhóm nghề thực tế (ground-truth do người dùng xác nhận).
  3. Ghép các mẫu dữ liệu mới này vào tập dữ liệu huấn luyện (loại bỏ trùng lặp) và gắn
     label_source = "feedback" để phân biệt với nhãn suy ra từ bảng điểm (label_source = "derived").
  4. Huấn luyện lại ĐÚNG LOẠI mô hình đang triển khai (nhân bản pipeline trong models/current/ qua
     sklearn.base.clone; mặc định Logistic Regression nếu chưa có mô hình) và lưu đè vào models/current/.
     Nhờ vậy thuật toán được chọn bởi train.py không bị âm thầm đổi khi huấn luyện lại.
  5. Kích hoạt cbf_recommender.reload() để cập nhật lại phản hồi của CBF (mốc chuẩn hoá z-score của CBF
     chỉ tính trên các dòng "derived" nên không bị dịch chuyển bởi dòng phản hồi).
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cbf_recommender import cbf_recommender
from .domain_knowledge import (
    LABEL_DERIVED, LABEL_FEEDBACK, LABEL_SOURCE_COL, SUBJECT_KEYS, SUBJECTS,
)

DATA_PATH = "data/students.csv"
FEEDBACK_PATH = "data/feedback.json"
CAREERS_PATH = "data/careers.json"
MODEL_PATH = os.path.join("models", "current", "model_pipeline.joblib")
FEATURE_COLS = [SUBJECT_KEYS[s] for s in SUBJECTS]
TARGET_COL = "career_group"
MAX_F1_DROP = 0.03  # cho phép F1 CV giảm tối đa 3 điểm % so với bản đang chạy


def _default_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def _current_pipeline() -> Pipeline:
    """Pipeline (chưa fit) cùng loại với mô hình đang triển khai; mặc định Logistic Regression."""
    if os.path.exists(MODEL_PATH):
        try:
            return clone(joblib.load(MODEL_PATH))
        except Exception:
            pass
    return _default_pipeline()


def retrain_from_explicit_feedback() -> dict:
    """
    Huấn luyện lại mô hình Tầng 1 (cùng thuật toán đang triển khai) và cập nhật hệ thống CBF từ phản hồi tường minh.
    """
    if not os.path.exists(FEEDBACK_PATH):
        return {
            "status": "warning",
            "samples_added": 0,
            "message": "Chưa có file phản hồi người dùng nào (data/feedback.json).",
        }

    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        feedbacks = json.load(f)

    with open(CAREERS_PATH, encoding="utf-8") as f:
        careers_meta = json.load(f)

    # 1. Trích xuất các mẫu học mới từ phản hồi tường minh
    new_rows = []
    for idx, fb in enumerate(feedbacks, 1):
        scores = fb.get("scores")
        liked = fb.get("liked_careers", [])

        if not scores or not liked or len(scores) < len(SUBJECTS):
            continue

        # Lấy nhóm nghề của các nghề được người dùng like
        confirmed_groups = set()
        for cid in liked:
            if cid in careers_meta:
                grp = careers_meta[cid].get("career_group")
                if grp:
                    confirmed_groups.add(grp)

        # Với mỗi nhóm nghề được người dùng xác nhận qua like
        for grp in confirmed_groups:
            row = {"name": f"Phản hồi #{idx} ({fb.get('rating', 5)}★)"}
            for s in SUBJECTS:
                row[SUBJECT_KEYS[s]] = float(scores.get(s, 7.0))
            row[TARGET_COL] = grp
            row[LABEL_SOURCE_COL] = LABEL_FEEDBACK
            new_rows.append(row)

    # 2. Đọc tập dữ liệu hiện tại
    df_base = pd.read_csv(DATA_PATH)
    if LABEL_SOURCE_COL not in df_base.columns:
        df_base[LABEL_SOURCE_COL] = LABEL_DERIVED
    df_base[LABEL_SOURCE_COL] = df_base[LABEL_SOURCE_COL].fillna(LABEL_DERIVED)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Hợp nhất và loại bỏ các dòng bị trùng lặp hoàn toàn về điểm và nhóm (giữ dòng gốc, đứng trước)
        df_merged = pd.concat([df_base, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
        samples_added = len(df_merged) - len(df_base)
    else:
        df_merged = df_base
        samples_added = 0

    X = df_merged[FEATURE_COLS]
    y = df_merged[TARGET_COL]

    # 3. Huấn luyện lại cùng loại mô hình đang triển khai
    pipeline = _current_pipeline()
    algo_name = type(pipeline.named_steps["model"]).__name__
    from .domain_knowledge import MODEL_LABELS
    full_algo_name = MODEL_LABELS.get(algo_name, algo_name)

    # Đánh giá 5-fold CV (độ khớp với nhãn suy ra + nhãn phản hồi, KHÔNG phải độ chính xác dự đoán thực tế)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_arr = cross_val_score(clone(pipeline), X, y, cv=skf, scoring="f1_weighted")
    cv_score, cv_std = float(cv_arr.mean()), float(cv_arr.std())

    # Cổng chất lượng: không triển khai model mới nếu F1 CV giảm quá MAX_F1_DROP so với phiên bản đang chạy.
    try:
        from .model_exporter import get_current_metadata as _gcm
        prev_f1 = _gcm().get("evaluation_metrics", {}).get("f1_cv_5fold_mean")
    except Exception:
        prev_f1 = None
    if isinstance(prev_f1, (int, float)) and cv_score < prev_f1 - MAX_F1_DROP:
        return {
            "status": "rejected",
            "samples_added": 0,
            "message": f"Từ chối triển khai: F1 5-fold CV mới {cv_score*100:.1f}% thấp hơn bản hiện tại "
                       f"{prev_f1*100:.1f}% quá {MAX_F1_DROP*100:.0f} điểm. Giữ nguyên model cũ.",
        }

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    # Cập nhật lại file data/students.csv (luôn ghi để bổ sung cột label_source nếu file cũ chưa có)
    df_merged.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    # GIAI ĐOẠN 2: EXPORT — Cập nhật metadata và tăng phiên bản model
    from .model_exporter import export_model_package, get_current_metadata
    curr_meta = get_current_metadata()
    old_version = curr_meta.get("model_version", "1.0.0")
    try:
        v_parts = old_version.split(".")
        new_version = f"{v_parts[0]}.{v_parts[1]}.{int(v_parts[2]) + 1}"
    except Exception:
        new_version = f"{old_version}.1"

    # Chỉ ghi các chỉ số thật sự đo được (F1 CV + độ lệch chuẩn thật); các chỉ số test không đo lại thì để None
    metrics_update = {
        "accuracy": None, "precision": None, "recall": None, "f1_test": None,
        "f1_cv_5fold_mean": float(round(cv_score, 4)),
        "f1_cv_5fold_std": float(round(cv_std, 4)),
    }
    meta = export_model_package(
        pipeline=pipeline,
        algorithm_name=full_algo_name,
        metrics=metrics_update,
        training_df=df_merged,
        version=new_version,
        model_path=MODEL_PATH,
    )

    # 4. Tải lại Content-Based Filtering Recommender
    cbf_recommender.reload()

    return {
        "status": "success",
        "samples_added": samples_added,
        "total_samples": len(df_merged),
        "algorithm": algo_name,
        "model_version": new_version,
        "f1_score": round(cv_score, 4),
        "message": f"Huấn luyện lại thành công ({algo_name})! Đã tích hợp {samples_added} mẫu từ phản hồi tường minh. "
                   f"F1 (5-fold CV, khớp với nhãn huấn luyện) = {cv_score*100:.1f}%.",
    }
