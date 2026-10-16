# -*- coding: utf-8 -*-
"""
Kiểm thử hồi quy cho 6 điểm không nhất quán đã được sửa. Chạy từ thư mục gốc dự án:

    python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

import joblib  # noqa: E402
from career_ai.cbf_recommender import cbf_recommender  # noqa: E402
from career_ai.domain_knowledge import (  # noqa: E402
    CAREER_GROUPS, SUBJECTS, TOP_N_CAREERS,
)

SCORES = {s: 7.5 for s in SUBJECTS}
SCORES.update({"Trí tuệ nhân tạo": 9.5, "Hệ quản trị CSDL": 9.0, "Xử lý ảnh": 8.5})


class TestSystem(unittest.TestCase):
    # ---- Điểm 1: mô tả thuật toán khớp với code ----
    def test_deployed_model_matches_docs(self):
        model = joblib.load(os.path.join(ROOT, "models", "current", "model_pipeline.joblib"))
        step = model.steps[-1][1]
        self.assertEqual(type(step).__name__, "LogisticRegression")
        readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        self.assertNotIn("Random Forest (triển khai)", readme)
        self.assertNotIn("lưu Random Forest", readme)

    def test_train_marks_only_best_model(self):
        tmp = tempfile.mkdtemp()
        try:
            shutil.copytree(os.path.join(ROOT, "career_ai"), os.path.join(tmp, "career_ai"))
            shutil.copytree(os.path.join(ROOT, "data"), os.path.join(tmp, "data"))
            out = subprocess.run(
                [sys.executable, "-m", "career_ai.train"], cwd=tmp, capture_output=True,
                text=True, encoding="utf-8",
            ).stdout
            marked = [l for l in out.splitlines() if "<-- triển khai" in l]
            self.assertEqual(len(marked), 1)
            self.assertIn("Logistic Regression", marked[0])
            self.assertIn("KIỂM ĐỊNH ĐỘC LẬP", out)
        finally:
            shutil.rmtree(tmp)

    def test_retrain_keeps_algorithm_and_labels_source(self):
        tmp = tempfile.mkdtemp()
        try:
            shutil.copytree(os.path.join(ROOT, "career_ai"), os.path.join(tmp, "career_ai"))
            shutil.copytree(os.path.join(ROOT, "data"), os.path.join(tmp, "data"))
            shutil.copytree(os.path.join(ROOT, "models", "current"), os.path.join(tmp, "models", "current"))
            fake = [{"id": "t1", "student_name": "Test", "rating": 4, "liked_careers": ["AI01"],
                     "disliked_careers": [], "scores": {s: 6.1 + i * 0.3 for i, s in enumerate(SUBJECTS)}}]
            with open(os.path.join(tmp, "data", "feedback.json"), "w", encoding="utf-8") as fh:
                json.dump(fake, fh, ensure_ascii=False)
            code = ("import sys; sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None;"
                    "import warnings; warnings.filterwarnings('ignore');"
                    "from career_ai.retrain_service import retrain_from_explicit_feedback as r; import json;"
                    "print('RES=' + json.dumps(r(), ensure_ascii=False))")
            sub_env = dict(os.environ, PYTHONIOENCODING="utf-8")
            out = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True, encoding="utf-8", env=sub_env)
            self.assertIn("RES=", out.stdout, out.stderr[-500:])
            self.assertIn("LogisticRegression", out.stdout)
            df = pd.read_csv(os.path.join(tmp, "data", "students.csv"))
            self.assertIn("label_source", df.columns)
            self.assertEqual(set(df["label_source"]), {"derived", "feedback"})
        finally:
            shutil.rmtree(tmp)

    # ---- Điểm 2: nhãn suy ra được đánh dấu rõ, không trộn với nhãn người dùng ----
    def test_students_csv_labels_are_marked_derived(self):
        df = pd.read_csv(os.path.join(ROOT, "data", "students.csv"))
        self.assertIn("label_source", df.columns)
        self.assertTrue((df["label_source"] == "derived").all())

    # ---- Điểm 3: hồ sơ nghề (tri thức chuyên gia) đầy đủ & báo cáo được điểm lệch ----
    def test_profile_consistency_has_no_config_errors(self):
        from career_ai.eval_system import check_profile_consistency
        problems, mismatches, total = check_profile_consistency()
        self.assertEqual(problems, [])
        self.assertEqual(total, 21)
        self.assertEqual(mismatches, [])  # 21/21 nghề nhất quán (MB01 đã được sửa hồ sơ)

    # ---- Điểm 4: kỹ năng chưa đánh giá không được cộng điểm ----
    def test_unassessed_skills_add_nothing(self):
        base = cbf_recommender.predict_career_cbf_scores(SCORES, None, "Dữ liệu & AI")
        empty = cbf_recommender.predict_career_cbf_scores(SCORES, {}, "Dữ liệu & AI")
        medium = cbf_recommender.predict_career_cbf_scores(
            SCORES, {"problem_solving": "medium", "english": "medium"}, "Dữ liệu & AI")
        self.assertEqual(base, empty)
        self.assertNotEqual(base, medium)  # "Trung bình" chọn có chủ đích vẫn có tác dụng

    def test_demo_feedback_record_removed(self):
        fb = json.load(open(os.path.join(ROOT, "data", "feedback.json"), encoding="utf-8"))
        for r in fb:
            sc = r.get("scores", {})
            self.assertFalse(sc.get("Công nghệ phần mềm") == 9.7 and sc.get("Trí tuệ nhân tạo") == 9.4
                             and sc.get("Kỹ thuật đồ hoạ máy tính") == 10.0, "bản ghi demo trùng bộ điểm cứng cũ")

    # ---- CBF theo sơ đồ: vector nghề thang 5-10 ----
    def test_career_profile_vectors_follow_diagram(self):
        from career_ai.domain_knowledge import CAREER_SUBJECT_WEIGHTS, career_profile_vector
        for cid, w in CAREER_SUBJECT_WEIGHTS.items():
            v = career_profile_vector(cid)
            self.assertEqual(len(v), len(SUBJECTS))
            self.assertTrue(all(5.0 <= x <= 10.0 for x in v))
            for i, s in enumerate(SUBJECTS):
                if w.get(s, 0) == 0:
                    self.assertEqual(v[i], 5.0)      # môn không liên quan = 5
                if w.get(s, 0) == 3:
                    self.assertEqual(v[i], 10.0)     # môn cốt lõi = 10
        mb = career_profile_vector("MB01")
        self.assertEqual(mb[SUBJECTS.index("Công nghệ web")], 5.0)   # MB01 không còn nghiêng về Web

    def test_profiles_are_distinct_within_group(self):
        from career_ai.domain_knowledge import career_profile_vector
        import json as _j
        careers = _j.load(open(os.path.join(ROOT, "data", "careers.json"), encoding="utf-8"))
        vecs = [tuple(career_profile_vector(c)) for c in careers]
        self.assertEqual(len(set(vecs)), len(vecs), "không có 2 nghề trùng vector (sẽ bằng điểm)")

    # ---- Điểm 6: luôn Top 3 nghề ----
    def test_recommend_top_careers_returns_exactly_three(self):
        for top_group in CAREER_GROUPS:
            others = [(g, 0.1) for g in CAREER_GROUPS if g != top_group]
            ranked = [(top_group, 0.5)] + others
            top, remaining = cbf_recommender.recommend_top_careers(ranked, SCORES, {})
            ids = [c["career_id"] for c in top]
            self.assertEqual(len(top), TOP_N_CAREERS, top_group)
            self.assertEqual(len(set(ids)), len(ids), "không được trùng nghề")
            self.assertEqual([c["rank"] for c in top], [1, 2, 3])
            self.assertFalse(set(ids) & {c["career_id"] for c in remaining})
            n_in_group = sum(1 for c in top if not c["is_group_fill"])
            self.assertEqual(n_in_group, min(3, len(cbf_recommender.rank_careers_in_group(top_group, SCORES))))

if __name__ == "__main__":
    unittest.main(verbosity=2)
