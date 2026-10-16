# -*- coding: utf-8 -*-
"""
api/main.py — Bước 2: Xây dựng REST API bằng FastAPI (PDF Mục 3.4).

API có các nhiệm vụ chính:
1. Nạp model một lần ở cấp module khi khởi động (không nạp lại trong mỗi request).
2. Đọc metadata cấu hình từ models/current/metadata.json.
3. Khai báo và kiểm tra dữ liệu đầu vào bằng Pydantic (ràng buộc 12 môn [0.0, 10.0], từ chối lỗi với mã 422).
4. Các endpoints:
   - GET  /health      : Sức khỏe dịch vụ, model version, uptime.
   - GET  /model-info  : Cung cấp metadata, schema đầu vào, danh mục môn học, nhóm nghề để Web render động.
   - POST /predict     : Suy luận phân loại nhóm nghề (ML Inference thuần túy).
   - POST /decision    : Hệ hỗ trợ quyết định toàn diện (ML Tầng 1 + CBF Tầng 2 + Gemini LLM diễn giải).
5. Kích hoạt CORSMiddleware để hỗ trợ giao diện Web Streamlit.
"""
import os
import sys
import time
import json
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd
import numpy as np
import joblib

# Thiết lập đường dẫn thư mục gốc
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "api" else CURRENT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Các module dùng chung (CBF, feedback_store, monitor_service...) đọc data/ bằng đường dẫn tương đối so với thư mục gốc.
# Chuyển thư mục làm việc về gốc để API chạy đúng dù khởi động từ api/ (như PDF) hay từ thư mục gốc.
os.chdir(PROJECT_ROOT)

from career_ai.domain_knowledge import (
    SUBJECTS, SUBJECT_KEYS, KEY_TO_SUBJECT, CAREER_GROUPS,
    SOFT_SKILL_OPTIONS, INTEREST_OPTIONS, SKILL_LEVEL_CHOICES, SKILL_LEVEL_LABELS,
)
from career_ai.cbf_recommender import ContentBasedFilteringRecommender
from api.llm_explain import DISCLAIMER, explain_with_llm
from career_ai.decision_core import apply_interest_and_skill_boost, clean_skill_levels
from career_ai.model_exporter import (
    METADATA_FILE, MODEL_FILE, sklearn_version_warning, validate_deployable_pipeline,
)
from career_ai.monitor_service import log_prediction

START_TIME = time.time()

# ---------------------------------------------------------------------------
# 1. NẠP MODEL VÀ METADATA MỘT LẦN Ở CẤP MODULE
# ---------------------------------------------------------------------------
# NGUỒN SỰ THẬT DUY NHẤT: models/current/ chứa pipeline đã triển khai và metadata.
# models/v<phiên bản>/ chỉ là bản lưu trữ lịch sử, không được nạp ở đây.
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_FILE)
METADATA_PATH = os.path.join(PROJECT_ROOT, METADATA_FILE)

if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    raise RuntimeError(
        "Chưa có models/current/model_pipeline.joblib / metadata.json. "
        "Hãy chạy `python -m career_ai.prepare_data` rồi `python -m career_ai.train` trước khi khởi động API."
    )

print(f"[API] Đang nạp model từ: {MODEL_PATH}")
pipe = validate_deployable_pipeline(joblib.load(MODEL_PATH))

print(f"[API] Đang nạp metadata từ: {METADATA_PATH}")
with open(METADATA_PATH, encoding="utf-8") as f:
    META = json.load(f)

_ver_warn = sklearn_version_warning(META)
if _ver_warn:
    print(f"[API][CẢNH BÁO] {_ver_warn}")

# Metadata dùng schema của model_exporter.py: model_version, input_schema, target_schema, evaluation_metrics.
MODEL_VERSION = META.get("model_version", "unknown")
ALGORITHM = META.get("algorithm", "unknown")
FEATURE_COLS = META.get("input_schema", {}).get("feature_order") or [SUBJECT_KEYS[s] for s in SUBJECTS]
FEATURE_RANGES = {
    f["feature_key"]: [f["min_value"], f["max_value"]]
    for f in META.get("input_schema", {}).get("features", [])
} or {c: [0.0, 10.0] for c in FEATURE_COLS}
CAREER_GROUP_LIST = META.get("target_schema", {}).get("classes") or CAREER_GROUPS
_ev = META.get("evaluation_metrics", {})
# Giữ tên khóa cũ (f1_score, f1_cv_5fold) để giao diện Web không phải sửa.
METRICS_PUBLIC = {
    "accuracy": _ev.get("accuracy"),
    "precision": _ev.get("precision"),
    "recall": _ev.get("recall"),
    "f1_score": _ev.get("f1_test"),
    "f1_cv_5fold": _ev.get("f1_cv_5fold_mean"),
    "f1_cv_5fold_std": _ev.get("f1_cv_5fold_std"),
}
METRICS_NOTE = (
    "Các chỉ số này đo mức mô hình khớp với nhãn nhóm nghề được SUY RA từ bảng điểm bằng quy tắc chuyên gia; "
    "chúng không phải độ chính xác dự đoán nghề nghiệp thực tế."
)

recommender = ContentBasedFilteringRecommender(
    students_csv_path=os.path.join(PROJECT_ROOT, "data", "students.csv"),
    careers_json_path=os.path.join(PROJECT_ROOT, "data", "careers.json"),
    feedback_json_path=os.path.join(PROJECT_ROOT, "data", "feedback.json"),
)


# ---------------------------------------------------------------------------
# 2. KHAI BÁO VÀ KIỂM TRA DỮ LIỆU ĐẦU VÀO BẰNG PYDANTIC
# ---------------------------------------------------------------------------
class SubjectScoresInput(BaseModel):
    """Điểm 12 môn học bắt buộc trong khoảng [0.0, 10.0]."""
    sec: float = Field(..., ge=0.0, le=10.0, description="An toàn và bảo mật HTTT")
    se: float = Field(..., ge=0.0, le=10.0, description="Công nghệ phần mềm")
    linux: float = Field(..., ge=0.0, le=10.0, description="Hệ điều hành Linux")
    db: float = Field(..., ge=0.0, le=10.0, description="Hệ quản trị CSDL")
    graphics: float = Field(..., ge=0.0, le=10.0, description="Kỹ thuật đồ hoạ máy tính")
    network: float = Field(..., ge=0.0, le=10.0, description="Mạng máy tính")
    analysis: float = Field(..., ge=0.0, le=10.0, description="Phân tích thiết kế HTTT")
    image: float = Field(..., ge=0.0, le=10.0, description="Xử lý ảnh")
    java: float = Field(..., ge=0.0, le=10.0, description="Công nghệ Java")
    csharp: float = Field(..., ge=0.0, le=10.0, description="Ngôn ngữ C# và công nghệ .NET")
    web: float = Field(..., ge=0.0, le=10.0, description="Công nghệ web")
    ai: float = Field(..., ge=0.0, le=10.0, description="Trí tuệ nhân tạo")


class PredictRequest(BaseModel):
    """Hỗ trợ cả định dạng phẳng lẫn bọc trong object 'scores' hoặc tên môn tiếng Việt."""
    scores: Optional[Dict[str, float]] = None
    sec: Optional[float] = Field(None, ge=0.0, le=10.0)
    se: Optional[float] = Field(None, ge=0.0, le=10.0)
    linux: Optional[float] = Field(None, ge=0.0, le=10.0)
    db: Optional[float] = Field(None, ge=0.0, le=10.0)
    graphics: Optional[float] = Field(None, ge=0.0, le=10.0)
    network: Optional[float] = Field(None, ge=0.0, le=10.0)
    analysis: Optional[float] = Field(None, ge=0.0, le=10.0)
    image: Optional[float] = Field(None, ge=0.0, le=10.0)
    java: Optional[float] = Field(None, ge=0.0, le=10.0)
    csharp: Optional[float] = Field(None, ge=0.0, le=10.0)
    web: Optional[float] = Field(None, ge=0.0, le=10.0)
    ai: Optional[float] = Field(None, ge=0.0, le=10.0)

    @model_validator(mode="before")
    @classmethod
    def validate_and_unify_scores(cls, data):
        if not isinstance(data, dict):
            return data
        
        # Nếu đã có trường scores bọc bên ngoài
        scores_dict = data.get("scores")
        flat_data = {k: v for k, v in data.items() if k != "scores"}
        
        extracted = {}
        # Ưu tiên lấy từ scores nếu có
        if isinstance(scores_dict, dict):
            for k, v in scores_dict.items():
                norm_k = SUBJECT_KEYS.get(k, k)
                extracted[norm_k] = v
        # Lấy bổ sung từ flat_data
        for k, v in flat_data.items():
            norm_k = SUBJECT_KEYS.get(k, k)
            if norm_k in FEATURE_COLS and v is not None:
                extracted[norm_k] = v

        # Kiểm tra đầy đủ 12 môn
        missing = [f for f in FEATURE_COLS if f not in extracted]
        if missing:
            missing_names = [KEY_TO_SUBJECT.get(m, m) for m in missing]
            raise ValueError(f"Thiếu điểm các môn học: {', '.join(missing_names)}")

        # Kiểm tra khoảng giá trị [0.0, 10.0]
        for f in FEATURE_COLS:
            val = extracted[f]
            if isinstance(val, bool):
                raise ValueError(f"Điểm môn '{KEY_TO_SUBJECT.get(f, f)}' phải là số, không phải true/false")
            try:
                fval = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Điểm môn '{KEY_TO_SUBJECT.get(f, f)}' phải là số hợp lệ, nhận được: {val}")
            if not math.isfinite(fval):  # NaN/Infinity lọt qua phép so sánh khoảng và gây lỗi 500
                raise ValueError(f"Điểm môn '{KEY_TO_SUBJECT.get(f, f)}' phải là số hữu hạn")
            if fval < 0.0 or fval > 10.0:
                raise ValueError(f"Điểm môn '{KEY_TO_SUBJECT.get(f, f)}' phải nằm trong khoảng [0.0, 10.0], nhận được: {fval}")
            extracted[f] = fval

        data["scores"] = extracted
        return data


class DecisionRequest(PredictRequest):
    """Request đầy đủ cho hệ hỗ trợ quyết định: điểm 12 môn, kỹ năng mềm, sở thích."""
    skills: Optional[Dict[str, str]] = Field(default_factory=dict, max_length=20, description="Kỹ năng mềm")
    interests: Optional[List[str]] = Field(default_factory=list, max_length=20, description="Lĩnh vực quan tâm")
    student_name: Optional[str] = Field(default="Sinh viên", max_length=100, description="Họ tên sinh viên")

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, v):
        """Khóa kỹ năng phải có trong danh mục; cấp độ phải là weak/medium/strong/excellent hoặc rỗng (= chưa đánh giá)."""
        valid_keys = {o["key"] for o in SOFT_SKILL_OPTIONS}
        for k, lvl in (v or {}).items():
            if k not in valid_keys:
                raise ValueError(f"Kỹ năng không hợp lệ: '{k}'. Hợp lệ: {sorted(valid_keys)}")
            if lvl not in SKILL_LEVEL_LABELS and lvl not in ("", "unassessed"):
                raise ValueError(f"Cấp độ kỹ năng '{lvl}' không hợp lệ. Hợp lệ: {sorted(SKILL_LEVEL_LABELS)} hoặc rỗng")
        return v or {}

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, v):
        bad = [i for i in (v or []) if i not in INTEREST_OPTIONS]
        if bad:
            raise ValueError(f"Lĩnh vực quan tâm không hợp lệ: {bad}. Hợp lệ: {INTEREST_OPTIONS}")
        return v or []


# ---------------------------------------------------------------------------
# 3. FASTAPI APP KHỞI TẠO VÀ CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Career Recommendation DSS API",
    description="Hệ thống hỗ trợ quyết định định hướng nghề nghiệp CNTT dựa trên mô hình học máy và lọc dựa trên nội dung (CBF).",
    version=MODEL_VERSION,
)

# Danh sách nguồn được phép, cách nhau bằng dấu phẩy, cấu hình qua biến môi trường CORS_ORIGINS.
# Mặc định chỉ cho phép localhost; không dùng "*" cùng allow_credentials=True (không hợp lệ và thiếu an toàn).
_cors_origins = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:5000,http://127.0.0.1:5000"
).split(",") if o.strip()]

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Trả 422 gọn, KHÔNG phản chiếu lại 'input' (NaN/Infinity làm JSON encode lỗi -> 500; input dài làm phình phản hồi)."""
    errors = [{"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")), "type": e.get("type", "")}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predict_raw(scores_dict: Dict[str, float]) -> dict:
    """Gửi trực tiếp 12 điểm thô vào Pipeline đã đóng gói cả tiền xử lý và model."""
    row = {k: scores_dict[k] for k in FEATURE_COLS}
    X = pd.DataFrame([row])[FEATURE_COLS]
    
    pred_group = str(pipe.predict(X)[0])
    classes = list(pipe.classes_)
    probas = pipe.predict_proba(X)[0]
    
    prob_dict = {cls_name: round(float(p) * 100, 2) for cls_name, p in zip(classes, probas)}
    # Sắp xếp các nhóm nghề theo xác suất giảm dần
    sorted_groups = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    confidence = round(float(prob_dict.get(pred_group, max(probas) * 100)), 2)

    return {
        "predicted_group": pred_group,
        "confidence": confidence,
        "probabilities": prob_dict,
        "sorted_groups": sorted_groups,
        "model_version": MODEL_VERSION,
        "algorithm": ALGORITHM,
    }


# ---------------------------------------------------------------------------
# 4. ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Kiểm tra tình trạng hoạt động của API và phiên bản mô hình."""
    uptime = round(time.time() - START_TIME, 2)
    return {
        "status": "healthy",
        "service": "Career Recommendation AI Serving API",
        "model_version": MODEL_VERSION,
        "algorithm": ALGORITHM,
        "uptime_seconds": uptime,
    }


@app.get("/model-info")
def model_info():
    """Trả về metadata để client dựng form nhập liệu và kiểm tra đầu vào."""
    return {
        "model_name": META.get("model_name", "career_recommendation_model"),
        "version": MODEL_VERSION,
        "algorithm": ALGORITHM,
        "numeric_features": FEATURE_COLS,
        "subject_names": SUBJECTS,
        "subject_keys": SUBJECT_KEYS,
        "key_to_subject": KEY_TO_SUBJECT,
        "feature_ranges": FEATURE_RANGES,
        "career_groups": CAREER_GROUP_LIST,
        "soft_skills": SOFT_SKILL_OPTIONS,
        "skill_levels": SKILL_LEVEL_CHOICES,
        "interest_options": INTEREST_OPTIONS,
        "metrics_test": METRICS_PUBLIC,
        "metrics_note": METRICS_NOTE,
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Endpoint /predict: Suy luận phân loại nhóm nghề (ML Tầng 1).
    Nhận điểm 12 môn học -> kiểm tra hợp lệ -> trả về nhóm nghề và phân phối xác suất.
    """
    t0 = time.time()
    scores = req.scores
    result = _predict_raw(scores)
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    return result


@app.post("/decision")
def decision_endpoint(req: DecisionRequest):
    """
    Endpoint /decision: Logic hỗ trợ quyết định toàn diện (PDF Mục 3.4).
    Kết hợp:
      1. Dự đoán nhóm ngành hàng đầu bằng mô hình học máy (Tầng 1).
      2. Xếp hạng và đề xuất Top 3 nghề cụ thể trong 21 nghề IT bằng Content-Based Filtering (Tầng 2).
      3. Phân tích môn học thế mạnh, môn cần cải thiện và sự tương thích kỹ năng mềm.
      4. Tạo diễn giải tự nhiên thông minh từ Gemini API (Bước 4).
    """
    t0 = time.time()
    scores = req.scores
    skills = req.skills or {}
    interests = req.interests or []
    student_name = req.student_name or "Sinh viên"

    # Quy đổi điểm về dạng tên môn tiếng Việt để chuyển cho CBF
    subject_scores_vn = {KEY_TO_SUBJECT[k]: scores[k] for k in FEATURE_COLS}

    # 1. Dự đoán Tầng 1 (mô hình học máy) rồi điều chỉnh theo sở thích/kỹ năng mềm.
    ml_result = _predict_raw(scores)
    skill_levels = clean_skill_levels(skills)  # "Chưa đánh giá" bị bỏ qua, không cộng/trừ điểm
    raw_ranked = [(g, p / 100.0) for g, p in ml_result["sorted_groups"]]
    ranked_groups = apply_interest_and_skill_boost(raw_ranked, interests, skill_levels)
    adjusted = bool(interests or skill_levels)
    pred_group, top_prob = ranked_groups[0]
    confidence = round(top_prob * 100, 2)
    probabilities_adjusted = {g: round(p * 100, 2) for g, p in ranked_groups}

    # 2. Khuyến nghị Tầng 2: CBF
    top_careers, remaining = recommender.recommend_top_careers(
        ranked_groups_with_prob=ranked_groups,
        student_scores=subject_scores_vn,
        skills_selected=skill_levels,
        n=3,
    )

    # 3. Phân tích điểm mạnh & điểm cần cải thiện tổng thể
    sorted_scores = sorted(subject_scores_vn.items(), key=lambda kv: kv[1], reverse=True)
    strengths = [f"{s} ({val}đ)" for s, val in sorted_scores if val >= 7.5][:4]
    improvements = [f"{s} ({val}đ)" for s, val in sorted_scores[::-1] if val < 6.5][:3]

    # Định dạng kết quả Top 3 nghề
    formatted_top_careers = []
    for c in top_careers:
        formatted_top_careers.append({
            "career_id": c["career_id"],
            "career_name": c["career_name"],
            "career_group": c["career_group"],
            "match_percentage": c["cbf_score_pct"],
            "description": c["description"],
            "salary": c["salary"],
            "trend": c["trend"],
            "strengths": c.get("strengths", []),
            "improvements": c.get("improvements", []),
            "soft_skill_matches": c.get("soft_skill_matches", []),
            "is_group_fill": c.get("is_group_fill", False),
        })

    # Tóm tắt khuyến nghị
    top_career_name = formatted_top_careers[0]["career_name"] if formatted_top_careers else ""
    rec_summary = (
        f"Nhóm nghề khớp nhất với điểm của bạn: '{pred_group}' (xác suất {confidence:.1f}%"
        f"{', đã điều chỉnh theo sở thích/kỹ năng mềm' if adjusted else ''}). "
        f"Nghề khớp hồ sơ môn học nhất: {top_career_name} ({formatted_top_careers[0]['match_percentage']:.1f}%)."
        if formatted_top_careers else
        f"Nhóm nghề khớp nhất với điểm của bạn: '{pred_group}' (xác suất {confidence:.1f}%"
        f"{', đã điều chỉnh theo sở thích/kỹ năng mềm' if adjusted else ''})."
    )

    intermediate_decision = {
        "predicted_group": pred_group,
        "confidence": confidence,
        "top_careers": formatted_top_careers,
        "strengths": [s.split(" (")[0] for s in strengths],
        "improvements": [s.split(" (")[0] for s in improvements],
        "recommendation_summary": rec_summary,
    }

    # 4. Bước 4: Diễn giải bằng Gemini API
    student_meta = {
        "scores": subject_scores_vn,
        "skills": skills,
        "interests": interests,
        "student_name": student_name,
    }
    llm_res = explain_with_llm(student_meta, intermediate_decision)

    latency = round((time.time() - t0) * 1000, 2)

    # Giai đoạn 5 (Monitor): ghi nhật ký lượt dự đoán (đầu vào, nhóm dự đoán, độ trễ thật) để phát hiện drift.
    try:
        log_prediction(input_scores=scores, top_group=pred_group, top_prob=top_prob,
                       model_version=MODEL_VERSION, latency_ms=latency, source="fastapi")
    except OSError as e:  # lỗi ghi log không được làm hỏng phản hồi
        print(f"[API][CẢNH BÁO] Không ghi được log giám sát: {type(e).__name__}")

    return {
        "student_name": student_name,
        "predicted_group": pred_group,
        "confidence": confidence,
        "probabilities": probabilities_adjusted,
        "probabilities_model_only": ml_result["probabilities"],
        "model_predicted_group": ml_result["predicted_group"],
        "adjusted_by_preferences": adjusted,
        "top_careers": formatted_top_careers,
        "remaining_careers": [
            {
                "career_id": r["career_id"],
                "career_name": r["career_name"],
                "match_percentage": r["cbf_score_pct"],
                "salary": r["salary"],
            }
            for r in remaining
        ],
        "strengths": strengths,
        "improvements": improvements,
        "recommendation_summary": rec_summary,
        "explanation_llm": llm_res,
        "disclaimer": DISCLAIMER,
        "model_version": MODEL_VERSION,
        "response_time_ms": latency,
    }
