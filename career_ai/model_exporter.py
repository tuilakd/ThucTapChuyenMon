# -*- coding: utf-8 -*-
"""
model_exporter.py — Quản lý Giai đoạn 2 (EXPORT) trong vòng đời Model AI.

Nhiệm vụ:
  1. Lưu model và bước tiền xử lý ra file (.joblib).
  2. Xuất kèm file metadata chuẩn hóa (models/current/metadata.json) bao gồm:
     - Phiên bản mô hình (model_version: SemVer).
     - Thời gian xuất (export_timestamp).
     - Thuật toán & Pipeline steps.
     - Tên 12 cột đầu vào và tên môn học tương ứng.
     - Miền giá trị hợp lệ của từng đặc trưng (min 0.0, max 10.0).
     - Các chỉ số đánh giá (Accuracy, Precision, Recall, F1 test, F1 5-fold CV).
     - Thống kê phân bố baseline (mean, std) phục vụ phát hiện drift ở Giai đoạn 5.
  3. Cung cấp hàm kiểm tra tính hợp lệ của dữ liệu đầu vào (input validation).
"""
import datetime
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .domain_knowledge import (
    CAREER_GROUPS, KEY_TO_SUBJECT, SUBJECT_KEYS, SUBJECTS,
)

MODEL_FILE = os.path.join("models", "current", "model_pipeline.joblib")
METADATA_FILE = os.path.join("models", "current", "metadata.json")
MODELS_DIR = "models"


def _metric(v):
    """Làm tròn chỉ số; giữ None nếu chỉ số không được đo lại (không bịa số 0.0)."""
    return None if v is None else float(round(v, 4))


def validate_deployable_pipeline(pipeline) -> Pipeline:
    """Require preprocessing and inference to be packaged in one fitted sklearn Pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Model triển khai phải là sklearn Pipeline, không xuất model rời.")
    if "preprocess" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Pipeline phải chứa các bước 'preprocess' và 'model'.")
    estimator = pipeline.named_steps["model"]
    if not callable(getattr(estimator, "predict", None)) or not callable(
        getattr(estimator, "predict_proba", None)
    ):
        raise ValueError("Bước 'model' trong Pipeline phải hỗ trợ predict() và predict_proba().")
    return pipeline


def get_feature_schema() -> list:
    """Trả về định nghĩa schema chuẩn của 12 cột điểm đầu vào kèm miền giá trị."""
    schema = []
    for subj in SUBJECTS:
        key = SUBJECT_KEYS[subj]
        schema.append({
            "feature_key": key,
            "subject_name": subj,
            "data_type": "float",
            "min_value": 0.0,
            "max_value": 10.0,
            "required": True,
            "unit": "thang điểm 10",
        })
    return schema


def calculate_baseline_stats(df: pd.DataFrame) -> dict:
    """Tính toán mean và std cho 12 môn trên tập dữ liệu huấn luyện để làm mốc so sánh drift."""
    feature_cols = [SUBJECT_KEYS[s] for s in SUBJECTS]
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            stats[col] = {
                "mean": float(round(df[col].mean(), 3)),
                "std": float(round(df[col].std(), 3)),
                "min": float(round(df[col].min(), 3)),
                "max": float(round(df[col].max(), 3)),
                "subject_name": KEY_TO_SUBJECT.get(col, col),
            }
    return stats


def export_model_package(
    pipeline: Pipeline,
    algorithm_name: str,
    metrics: dict,
    training_df: pd.DataFrame,
    version: str = "1.0.0",
    model_path: str = MODEL_FILE,
    metadata_path: str = METADATA_FILE,
    archive: bool = True,
) -> dict:
    """
    Lưu pipeline tiền xử lý + mô hình cùng file metadata hoàn chỉnh.
    Hỗ trợ lưu trữ theo phiên bản trong thư mục models/v{version}/.
    """
    pipeline = validate_deployable_pipeline(pipeline)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)

    # 1. Lưu pipeline nhị phân
    joblib.dump(pipeline, model_path)

    # 2. Xây dựng metadata
    steps = [name for name, _ in pipeline.steps] if hasattr(pipeline, "steps") else [type(pipeline).__name__]
    classes = list(getattr(pipeline, "classes_", CAREER_GROUPS))

    baseline_stats = calculate_baseline_stats(training_df) if training_df is not None else {}
    total_samples = len(training_df) if training_df is not None else 0
    derived_samples = int((training_df["label_source"] == "derived").sum()) if (training_df is not None and "label_source" in training_df.columns) else total_samples
    feedback_samples = total_samples - derived_samples

    metadata = {
        "model_version": version,
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm_name,
        "framework": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "pipeline_steps": steps,
        "input_schema": {
            "feature_count": len(SUBJECTS),
            "feature_order": [SUBJECT_KEYS[s] for s in SUBJECTS],
            "features": get_feature_schema(),
        },
        "target_schema": {
            "target_column": "career_group",
            "class_count": len(classes),
            "classes": classes,
        },
        "evaluation_metrics": {
            "accuracy": _metric(metrics.get("accuracy")),
            "precision": _metric(metrics.get("precision")),
            "recall": _metric(metrics.get("recall")),
            "f1_test": _metric(metrics.get("f1_test")),
            "f1_cv_5fold_mean": _metric(metrics.get("f1_cv_5fold_mean")),
            "f1_cv_5fold_std": _metric(metrics.get("f1_cv_5fold_std")),
            "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        },
        "training_dataset_summary": {
            "total_samples": total_samples,
            "derived_samples": derived_samples,
            "feedback_samples": feedback_samples,
        },
        "baseline_feature_stats": baseline_stats,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Lưu thêm 1 bản snapshot vào thư mục models/v{version}/ nếu có cấu hình
    if archive:
        version_dir = os.path.join(MODELS_DIR, f"v{version}")
        os.makedirs(version_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(version_dir, "model_pipeline.joblib"))
        with open(os.path.join(version_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def load_model_with_metadata(
    model_path: str = MODEL_FILE,
    metadata_path: str = METADATA_FILE,
) -> tuple:
    """Nạp cả pipeline mô hình và file metadata kiểm định."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    pipeline = validate_deployable_pipeline(joblib.load(model_path))

    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    return pipeline, metadata


def sklearn_version_warning(metadata: dict | None) -> str | None:
    """Trả về thông báo nếu model được lưu bằng phiên bản scikit-learn khác phiên bản đang chạy."""
    saved = (metadata or {}).get("sklearn_version")
    if saved and saved != sklearn.__version__:
        return (f"Model được lưu bằng scikit-learn {saved} nhưng môi trường đang dùng {sklearn.__version__}. "
                f"Hãy chạy lại `python -m career_ai.train` để tạo model khớp phiên bản.")
    return None


def get_current_metadata(metadata_path: str = METADATA_FILE) -> dict:
    """Đọc file metadata hiện tại hoặc tạo metadata mặc định nếu chưa xuất."""
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_version": "1.0.0",
        "algorithm": "Logistic Regression (multinomial)",
        "input_schema": {"features": get_feature_schema()},
        "target_schema": {"classes": CAREER_GROUPS},
        "evaluation_metrics": {},
    }


def validate_input_scores(scores: dict) -> tuple[bool, str, dict]:
    """
    Kiểm tra tính hợp lệ của dữ liệu điểm 12 môn đầu vào:
      - Đủ 12 môn (hỗ trợ cả khoá tên tiếng Việt và khoá ngắn key).
      - Giá trị kiểu số và thuộc [0.0, 10.0].
    Trả về (is_valid, error_message, normalized_scores_dict_by_keys).
    """
    if not isinstance(scores, dict):
        return False, "Dữ liệu điểm phải là một đối tượng JSON/dict.", {}

    normalized = {}
    missing = []
    out_of_range = []

    for subj in SUBJECTS:
        key = SUBJECT_KEYS[subj]
        val = None
        if subj in scores:
            val = scores[subj]
        elif key in scores:
            val = scores[key]

        if val is None or str(val).strip() == "":
            missing.append(f"{subj} ({key})")
            continue

        try:
            val_f = float(val)
        except (ValueError, TypeError):
            return False, f"Điểm môn '{subj}' không phải số hợp lệ: {val}", {}

        if not (0.0 <= val_f <= 10.0):
            out_of_range.append(f"{subj}: {val_f}")

        normalized[key] = val_f

    if missing:
        return False, f"Thiếu điểm các môn học bắt buộc: {', '.join(missing)}", {}

    if out_of_range:
        return False, f"Điểm phải nằm trong khoảng [0.0, 10.0]. Điểm không hợp lệ: {', '.join(out_of_range)}", {}

    return True, "", normalized
