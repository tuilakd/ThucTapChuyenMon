# -*- coding: utf-8 -*-
"""
Kiểm thử parser bảng điểm, đầu vào bất thường, chỉ số và đường dẫn.
Chạy: python -m unittest discover -s tests -v
"""
import io
import os
import sys
import unittest
import warnings

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

KEYS = ["sec", "se", "linux", "db", "graphics", "network", "analysis", "image", "java", "csharp", "web", "ai"]
BASE_JSON = "{" + ", ".join(f'"{k}": 6.5' for k in KEYS) + "}"


@unittest.skipUnless(HAS_FASTAPI, "cần fastapi + httpx")
class TestApiHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["GEMINI_API_KEY"] = ""
        import api.main as api_main
        cls.client = TestClient(api_main.app)

    def _post(self, body: str):
        return self.client.post("/decision", content=body, headers={"Content-Type": "application/json"})

    def test_nan_and_infinity_rejected_with_422_not_500(self):
        for bad in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=bad):
                r = self._post(BASE_JSON.replace('"ai": 6.5', f'"ai": {bad}'))
                self.assertEqual(r.status_code, 422, r.text[:200])

    def test_boolean_score_rejected(self):
        r = self._post(BASE_JSON.replace('"ai": 6.5', '"ai": true'))
        self.assertEqual(r.status_code, 422)

    def test_overlong_name_rejected_and_input_not_echoed(self):
        r = self._post(BASE_JSON[:-1] + ', "student_name": "' + "x" * 5000 + '"}')
        self.assertEqual(r.status_code, 422)
        self.assertLess(len(r.text), 1000, "phản hồi lỗi không được phản chiếu lại input dài")

    def test_boundaries_accepted(self):
        for v in ("0.0", "10.0"):
            r = self.client.post("/predict", json={k: float(v) for k in KEYS})
            self.assertEqual(r.status_code, 200, v)


class TestUploadParsing(unittest.TestCase):
    def test_parse_uploaded_scores_handles_horizontal_and_vertical_excel_rows(self):
        import web.app as web_app

        subject_names = [
            "An toàn và bảo mật HTTT",
            "Công nghệ phần mềm",
            "Hệ điều hành Linux",
            "Hệ quản trị CSDL",
            "Kỹ thuật đồ hoạ máy tính",
            "Mạng máy tính",
            "Phân tích thiết kế HTTT",
            "Xử lý ảnh",
            "Công nghệ Java",
            "Ngôn ngữ C# và công nghệ .NET",
            "Công nghệ web",
            "Trí tuệ nhân tạo",
        ]
        subject_keys = {
            "An toàn và bảo mật HTTT": "sec",
            "Công nghệ phần mềm": "se",
            "Hệ điều hành Linux": "linux",
            "Hệ quản trị CSDL": "db",
            "Kỹ thuật đồ hoạ máy tính": "graphics",
            "Mạng máy tính": "network",
            "Phân tích thiết kế HTTT": "analysis",
            "Xử lý ảnh": "image",
            "Công nghệ Java": "java",
            "Ngôn ngữ C# và công nghệ .NET": "csharp",
            "Công nghệ web": "web",
            "Trí tuệ nhân tạo": "ai",
        }

        horizontal = pd.DataFrame([
            [8.0, 7.5, 9.0, 6.5, 7.0, 8.5, 7.8, 6.2, 9.3, 8.1, 8.8, 9.1]
        ], columns=list(subject_keys.values()))
        vertical = pd.DataFrame([
            ["An toàn và bảo mật HTTT", 8.0],
            ["Công nghệ phần mềm", 7.5],
            ["Hệ điều hành Linux", 9.0],
            ["Hệ quản trị CSDL", 6.5],
            ["Kỹ thuật đồ hoạ máy tính", 7.0],
            ["Mạng máy tính", 8.5],
            ["Phân tích thiết kế HTTT", 7.8],
            ["Xử lý ảnh", 6.2],
            ["Công nghệ Java", 9.3],
            ["Ngôn ngữ C# và công nghệ .NET", 8.1],
            ["Công nghệ web", 8.8],
            ["Trí tuệ nhân tạo", 9.1],
        ], columns=["Tên môn", "Điểm TBCHP"])

        self.assertEqual(len(web_app.parse_uploaded_scores(horizontal, subject_names, subject_keys)), 12)
        self.assertEqual(len(web_app.parse_uploaded_scores(vertical, subject_names, subject_keys)), 12)

    def test_parse_uploaded_workbook_with_title_row_course_codes_and_final_grade(self):
        import web.app as web_app

        subject_names = [
            "An toàn và bảo mật HTTT", "Công nghệ phần mềm", "Hệ điều hành Linux",
            "Hệ quản trị CSDL", "Kỹ thuật đồ hoạ máy tính", "Mạng máy tính",
            "Phân tích thiết kế HTTT", "Xử lý ảnh", "Công nghệ Java",
            "Ngôn ngữ C# và công nghệ .NET", "Công nghệ web", "Trí tuệ nhân tạo",
        ]
        subject_keys = {
            "An toàn và bảo mật HTTT": "sec", "Công nghệ phần mềm": "se",
            "Hệ điều hành Linux": "linux", "Hệ quản trị CSDL": "db",
            "Kỹ thuật đồ hoạ máy tính": "graphics", "Mạng máy tính": "network",
            "Phân tích thiết kế HTTT": "analysis", "Xử lý ảnh": "image",
            "Công nghệ Java": "java", "Ngôn ngữ C# và công nghệ .NET": "csharp",
            "Công nghệ web": "web", "Trí tuệ nhân tạo": "ai",
        }
        course_codes = [
            "TH 5210", "TH 4306", "TH 5211", "TH 5221", "TH 5231", "TH 5206",
            "TH 5208", "TH 5209", "TH 4316", "TH 4315", "TH 4309", "TH 4320",
        ]
        workbook = io.BytesIO()
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame([
                ["BẢNG ĐIỂM SINH VIÊN"],
                ["Mã học phần", "Tên học phần", "Số tín chỉ", "Điểm quá trình", "Điểm tổng kết"],
                *[
                    [code, subject, 3, 5.0, 7.0 + index / 10]
                    for index, (code, subject) in enumerate(zip(course_codes, subject_names))
                ],
            ]).to_excel(writer, sheet_name="Bảng điểm", header=False, index=False)
        workbook.seek(0)

        parsed, columns = web_app.parse_uploaded_file(
            workbook, "bang_diem.xlsx", subject_names, subject_keys
        )

        self.assertEqual(len(parsed), 12)
        self.assertEqual(columns, 5)
        self.assertEqual(parsed["sec"], 7.0)
        self.assertEqual(parsed["ai"], 8.1)


class TestConfig(unittest.TestCase):
    def test_unmeasured_metric_stays_none(self):
        from career_ai.model_exporter import _metric
        self.assertIsNone(_metric(None))
        self.assertEqual(_metric(0.123456), 0.1235)

    def test_default_data_paths_are_absolute(self):
        import inspect
        from career_ai.cbf_recommender import ContentBasedFilteringRecommender as C
        for name, p in inspect.signature(C.__init__).parameters.items():
            if name.endswith("_path"):
                self.assertTrue(os.path.isabs(p.default), f"{name} phải là đường dẫn tuyệt đối")

    def test_render_yaml_uses_hostport_for_api_url(self):
        src = open(os.path.join(ROOT, "render.yaml"), encoding="utf-8").read()
        self.assertIn("property: hostport", src)


if __name__ == "__main__":
    unittest.main()
