# -*- coding: utf-8 -*-
"""
Kiểm thử FastAPI /decision, logic dùng chung, validation, giám sát và cách khởi động theo tài liệu.
Chạy: python -m unittest discover -s tests -v
"""
import os
import subprocess
import sys
import tempfile
import unittest
import json
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from career_ai import decision_core, monitor_service  # noqa: E402

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

KEYS = ["sec", "se", "linux", "db", "graphics", "network", "analysis", "image", "java", "csharp", "web", "ai"]
BALANCED = {k: 6.5 for k in KEYS}
AI_STRONG = dict(BALANCED, db=9.0, ai=9.5, analysis=8.5, image=8.0)


class TestSharedLogic(unittest.TestCase):
    def test_unassessed_skills_are_dropped(self):
        cleaned = decision_core.clean_skill_levels({"teamwork": "unassessed", "english": "", "self_learning": "strong"})
        self.assertEqual(cleaned, {"self_learning": "strong"})

    def test_no_preferences_means_no_change_in_order(self):
        base = [("A", 0.5), ("B", 0.3), ("C", 0.2)]
        out = decision_core.apply_interest_and_skill_boost(base, [], {})
        self.assertEqual([g for g, _ in out], ["A", "B", "C"])
        self.assertAlmostEqual(sum(p for _, p in out), 1.0)


@unittest.skipUnless(HAS_FASTAPI, "cần fastapi + httpx")
class TestDecisionApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["GEMINI_API_KEY"] = ""
        from api.main import app as fastapi_app
        cls.api = TestClient(fastapi_app)
        cls._tmp = tempfile.TemporaryDirectory()
        cls._old_logs = monitor_service.LOGS_FILE
        monitor_service.LOGS_FILE = os.path.join(cls._tmp.name, "logs.jsonl")  # không ghi vào data/ thật

    @classmethod
    def tearDownClass(cls):
        monitor_service.LOGS_FILE = cls._old_logs
        cls._tmp.cleanup()

    def _api(self, scores, interests=(), skills=None):
        r = self.api.post("/decision", json=dict(scores, interests=list(interests), skills=skills or {}))
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_same_input_returns_consistent_group_and_careers(self):
        cases = [
            (BALANCED, [], {}),
            (AI_STRONG, [], {}),
            (BALANCED, ["Phát triển Web"], {}),
            (AI_STRONG, ["Đồ họa & Game"], {"self_learning": "excellent", "english": "strong"}),
        ]
        for scores, interests, skills in cases:
            with self.subTest(interests=interests, skills=skills):
                a = self._api(scores, interests, skills)
                b = self._api(scores, interests, skills)
                self.assertEqual(a["predicted_group"], b["predicted_group"])
                self.assertEqual(
                    [c["career_id"] for c in a["top_careers"]],
                    [c["career_id"] for c in b["top_careers"]],
                )

    def test_interest_changes_group_probability(self):
        base = self._api(BALANCED)
        boosted = self._api(BALANCED, ["Phát triển Web"])
        self.assertFalse(base["adjusted_by_preferences"])
        self.assertTrue(boosted["adjusted_by_preferences"])
        self.assertGreater(boosted["probabilities"]["Phát triển Web"], base["probabilities"]["Phát triển Web"])
        self.assertEqual(boosted["probabilities_model_only"], base["probabilities_model_only"])

    def test_unassessed_skill_equals_no_skill(self):
        a = self._api(AI_STRONG)
        b = self._api(AI_STRONG, skills={"teamwork": "unassessed", "english": ""})
        self.assertEqual(a["probabilities"], b["probabilities"])
        self.assertEqual([c["career_name"] for c in a["top_careers"]], [c["career_name"] for c in b["top_careers"]])

    def test_invalid_skill_and_interest_rejected(self):
        for payload in ({"skills": {"khong_ton_tai": "strong"}},
                        {"skills": {"teamwork": "sieu_gioi"}},
                        {"interests": ["Nghề không có thật"]}):
            with self.subTest(payload=payload):
                r = self.api.post("/decision", json=dict(BALANCED, **payload))
                self.assertEqual(r.status_code, 422)

    def test_monitor_log_has_real_latency(self):
        self._api(BALANCED)
        with open(monitor_service.LOGS_FILE, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        sources = {r["source"] for r in recs}
        self.assertIn("fastapi", sources)
        for r in recs:
            self.assertGreater(r["latency_ms"], 0)
            self.assertNotEqual(r["latency_ms"], 12.5, "độ trễ không được gán cứng")


@unittest.skipUnless(HAS_FASTAPI, "cần fastapi + httpx")
class TestRunFromApiDir(unittest.TestCase):
    def test_import_from_api_directory(self):
        """PDF chạy `cd api && uvicorn main:app`; module phải import được khi thư mục làm việc là api/."""
        env = dict(os.environ, GEMINI_API_KEY="", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "-c", "import main; print('VERSION=' + main.MODEL_VERSION)"],
                           cwd=os.path.join(ROOT, "api"), env=env, capture_output=True, text=True,
                           encoding="utf-8", timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("VERSION=", r.stdout)


if __name__ == "__main__":
    unittest.main()
