# -*- coding: utf-8 -*-
"""
test_lifecycle.py — Kiểm thử toàn diện 5 giai đoạn vòng đời Model AI trong ứng dụng:
  [1. Train] -> [2. Export] -> [3. Serve] -> [4. Integrate] -> [5. Monitor]
     ^                                                                |
     |_________________ Dữ liệu mới / Model lệch _____________________|
"""
import json
import os
import sys
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient
from api.main import app as api_app
from career_ai.domain_knowledge import SUBJECT_KEYS, SUBJECTS, CAREER_GROUPS
from career_ai.model_exporter import (
    METADATA_FILE, MODEL_FILE, get_current_metadata,
    validate_input_scores,
)
from career_ai.monitor_service import (
    log_prediction, get_prediction_logs, compute_data_drift,
    compute_prediction_drift,
)
from career_ai.retrain_service import retrain_from_explicit_feedback

SAMPLE_SCORES = {s: 8.0 for s in SUBJECTS}
SAMPLE_SCORES["Trí tuệ nhân tạo"] = 9.5
SAMPLE_SCORES["Hệ quản trị CSDL"] = 9.0


class TestModelLifecycle(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_app)

    # =========================================================================
    # GIAI ĐOẠN 1: TRAIN (Huấn luyện & Đánh giá)
    # =========================================================================
    def test_stage1_train_data_and_notebook_exist(self):
        """Kiểm tra dữ liệu huấn luyện và Jupyter Notebook EDA/Train đầy đủ."""
        self.assertTrue(os.path.exists("data/students.csv"), "Thiếu data/students.csv")
        self.assertTrue(os.path.exists("model/01_train_export_model.ipynb"), "Thiếu file Jupyter Notebook Giai đoạn 1")

        with open("model/01_train_export_model.ipynb", encoding="utf-8") as f:
            nb = json.load(f)
        self.assertIn("cells", nb)
        self.assertGreater(len(nb["cells"]), 5)

    # =========================================================================
    # GIAI ĐOẠN 2: EXPORT (Lưu model + metadata)
    # =========================================================================
    def test_stage2_export_metadata_schema_validity(self):
        """Kiểm tra metadata có đủ tên cột, miền giá trị [0, 10], metrics, version."""
        self.assertTrue(os.path.exists(MODEL_FILE), "Thiếu model file")
        self.assertTrue(os.path.exists(METADATA_FILE), "Thiếu models/current/metadata.json")

        meta = get_current_metadata()
        self.assertIn("model_version", meta)
        self.assertIn("input_schema", meta)
        self.assertIn("target_schema", meta)
        self.assertIn("evaluation_metrics", meta)

        features = meta["input_schema"]["features"]
        self.assertEqual(len(features), 12, "Cần định nghĩa đủ 12 môn học")
        for f in features:
            self.assertEqual(f["min_value"], 0.0)
            self.assertEqual(f["max_value"], 10.0)
            self.assertTrue(f["required"])

        classes = meta["target_schema"]["classes"]
        self.assertEqual(len(classes), 7, "Cần có 7 nhóm nghề mục tiêu")
        self.assertIn("f1_cv_5fold_mean", meta["evaluation_metrics"])

    def test_stage2_input_validation(self):
        """Kiểm tra cơ chế validate_input_scores theo schema đã export."""
        # 1. Điểm hợp lệ
        valid, msg, norm = validate_input_scores(SAMPLE_SCORES)
        self.assertTrue(valid)
        self.assertEqual(len(norm), 12)

        # 2. Thiếu môn
        incomplete = {s: 8.0 for s in SUBJECTS[:10]}
        valid, msg, _ = validate_input_scores(incomplete)
        self.assertFalse(valid)
        self.assertIn("Thiếu điểm", msg)

        # 3. Điểm ngoài miền [0, 10]
        out_of_bound = dict(SAMPLE_SCORES)
        out_of_bound["Công nghệ web"] = 11.5
        valid, msg, _ = validate_input_scores(out_of_bound)
        self.assertFalse(valid)
        self.assertIn("[0.0, 10.0]", msg)

    # =========================================================================
    # GIAI ĐOẠN 3: SERVE (Nạp model, mở REST API)
    # =========================================================================
    def test_stage3_api_health_endpoint(self):
        """Kiểm tra GET /health."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_version", data)
        self.assertIn("algorithm", data)

    def test_stage3_api_metadata_endpoint(self):
        """Kiểm tra GET /model-info."""
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["subject_names"]), 12)
        self.assertEqual(len(data["feature_ranges"]), 12)

    def test_stage3_api_predict_inference_endpoint(self):
        """Kiểm tra POST /predict suy luận thuần tuý."""
        # 1. Hợp lệ
        payload = {"scores": {SUBJECT_KEYS[s]: SAMPLE_SCORES[s] for s in SUBJECTS}}
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["predicted_group"], CAREER_GROUPS)
        self.assertEqual(len(data["probabilities"]), 7)
        self.assertGreater(data["confidence"], 0.0)

        # 2. Không hợp lệ (điểm âm)
        bad_payload = {"scores": dict(payload["scores"], ai=-1.0)}
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_stage3_api_recommend_endpoint(self):
        """Kiểm tra POST /decision (kết hợp ML Tầng 1 + CBF Tầng 2)."""
        payload = {SUBJECT_KEYS[s]: SAMPLE_SCORES[s] for s in SUBJECTS}
        payload.update(skills={"teamwork": "strong"}, interests=["Dữ liệu & AI"])
        resp = self.client.post("/decision", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["top_careers"]), 3)

    # =========================================================================
    # GIAI ĐOẠN 4: INTEGRATE (Tích hợp Web / Client)
    # =========================================================================
    def test_stage4_web_integration(self):
        """Kiểm tra Streamlit frontend và API cùng khai báo bộ nhập 12 môn."""
        from web import app as web_app
        self.assertTrue(os.path.exists("web/app.py"))
        self.assertTrue(callable(web_app.parse_uploaded_file))
        data = self.client.get("/model-info").json()
        self.assertEqual(len(data["subject_names"]), 12)

    # =========================================================================
    # GIAI ĐOẠN 5: MONITOR (Theo dõi, Drift, Feedback & Retrain Loop)
    # =========================================================================
    def test_stage5_prediction_logging_and_drift(self):
        """Kiểm tra ghi log dự đoán và tính toán Data Drift."""
        log = log_prediction(
            input_scores=SAMPLE_SCORES,
            top_group="Dữ liệu & AI",
            top_prob=0.88,
            model_version="1.0.0",
            latency_ms=8.5,
            source="test_runner",
        )
        self.assertIn("request_id", log)
        self.assertEqual(log["predicted_group"], "Dữ liệu & AI")

        # Kiểm tra đọc log
        logs = get_prediction_logs(limit=10)
        self.assertGreater(len(logs), 0)

        # Kiểm tra đo lường Data Drift
        drift = compute_data_drift()
        self.assertIn("has_drift", drift)
        self.assertIn("features", drift)
        self.assertEqual(len(drift["features"]), 12)

        # Kiểm tra đo lường Prediction Drift
        pred_drift = compute_prediction_drift()
        self.assertIn("distribution", pred_drift)

    def test_stage5_feedback_loop_retrain_updates_version(self):
        """
        Kiểm tra vòng lặp khép kín:
        [5. Monitor] -> [1. Train] -> [2. Export (tăng version)] -> [3. Serve]
        """
        initial_meta = get_current_metadata()
        v_init = initial_meta.get("model_version", "1.0.0")

        import shutil
        fb_path = "data/feedback.json"
        std_path = "data/students.csv"
        fb_bak = fb_path + ".bak"
        std_bak = std_path + ".bak"
        if os.path.exists(fb_path):
            shutil.copy(fb_path, fb_bak)
        if os.path.exists(std_path):
            shutil.copy(std_path, std_bak)

        try:
            # Giả lập feedback mới
            fake_fb = [{
                "id": "lifecycle-test-1",
                "student_name": "Lifecycle Tester",
                "rating": 5,
                "liked_careers": ["AI01"],
                "disliked_careers": [],
                "scores": {s: 8.5 for s in SUBJECTS},
            }]
            with open(fb_path, "w", encoding="utf-8") as f:
                json.dump(fake_fb, f, ensure_ascii=False)

            # Kích hoạt Retrain Loop
            res = retrain_from_explicit_feedback()
            self.assertEqual(res["status"], "success")
            self.assertIn("model_version", res)

            # Kiểm tra Giai đoạn 2 đã xuất version mới
            new_meta = get_current_metadata()
            self.assertNotEqual(new_meta["model_version"], v_init)
            self.assertEqual(new_meta["model_version"], res["model_version"])
        finally:
            if os.path.exists(fb_bak):
                shutil.move(fb_bak, fb_path)
            if os.path.exists(std_bak):
                shutil.move(std_bak, std_path)


if __name__ == "__main__":
    unittest.main()
