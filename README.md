# HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH ĐỊNH HƯỚNG NGHỀ NGHIỆP CNTT (AI CAREER DSS)
> **Ứng dụng tích hợp Model AI trên nền tảng Web theo chuẩn Bộ môn KHMT-CNPM**
> **Kiến trúc kết hợp:** Model Học máy tự huấn luyện (Tầng 1) + Lọc dựa trên nội dung CBF (Tầng 2) + Mô hình ngôn ngữ lớn Google Gemini 2.5 Flash (Bước 4)

---

## 1. Giới thiệu Bài toán & Mục tiêu

Sinh viên ngành Công nghệ Thông tin (CNTT) thường gặp khó khăn trong việc chọn hướng chuyên sâu phù hợp với năng lực học tập và thế mạnh cá nhân giữa rất nhiều định hướng nghề nghiệp khác nhau (AI, Dữ liệu, Web, Di động, An toàn thông tin, Hạ tầng mạng, Quản lý dự án,...).

Dự án này xây dựng một **Hệ Thống Hỗ Trợ Quyết Định (Decision Support System - DSS)** hoàn chỉnh, đưa model AI từ nghiên cứu ra ứng dụng thực tế trên nền tảng **Web** theo đúng quy trình 5 giai đoạn được hướng dẫn trong tài liệu:
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   1. Train   │ ──► │  2. Export   │ ──► │   3. Serve   │ ──► │ 4. Integrate │ ──► │  5. Monitor  │
│  Huấn luyện  │     │  Lưu model   │     │  Nạp model,  │     │ Tích hợp vào │     │ Theo dõi, thu│
│  (Jupyter)   │     │  + metadata  │     │  mở API/SDK  │     │     Web      │     │ thập feedback│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       ▲                                                                                   │
       └───────────────────────── Dữ liệu mới / Retrain Loop ──────────────────────────────┘
```

### Quy trình ra khuyến nghị (3 Tầng AI phối hợp):
1. **Tầng 1 — Phân loại Nhóm nghề (Machine Learning):** Điểm 12 môn học -> `StandardScaler` -> `Logistic Regression (multinomial)` -> Tính phân phối xác suất và độ tin cậy giữa 7 nhóm ngành CNTT lớn.
2. **Tầng 2 — Xếp hạng nghề nghiệp cụ thể (Content-Based Filtering - CBF):** Kết hợp Cosine Similarity, độ thuần thục môn cốt lõi (Core Subject Mastery), mức độ đáp ứng kỹ năng mềm và phản hồi cộng đồng để chọn ra **Top 3 nghề nghiệp triển vọng nhất** trong 21 nghề IT.
3. **Tầng 3 / Bước 4 — Diễn giải tự nhiên (Google Gemini API):** Gemini 2.5 Flash tiếp nhận toàn bộ số liệu định lượng từ Tầng 1 và Tầng 2 để tổng hợp lời tư vấn định hướng tự nhiên, khách quan bằng tiếng Việt mà không bịa số liệu.

---

## 2. Kiến trúc Triển khai (Server / REST API)

Hệ thống được tổ chức theo kiến trúc **Server / REST API** (Kiến trúc b trong tài liệu hướng dẫn):
- **Model Server (Backend):** Sử dụng **FastAPI**, nạp pipeline model một lần khi khởi động, kiểm tra dữ liệu bằng **Pydantic**, mở các RESTful endpoints:
  - `GET /health`: Kiểm tra sức khỏe hệ thống, phiên bản model, thời gian hoạt động.
  - `GET /model-info`: Cung cấp siêu dữ liệu (metadata), tên 12 môn học, khoảng điểm hợp lệ `[0.0, 10.0]`, danh mục kỹ năng để giao diện render động.
  - `POST /predict`: Suy luận ML thuần túy (nhóm nghề dự đoán, xác suất 7 nhóm).
  - `POST /decision`: Hệ hỗ trợ quyết định toàn diện (ML Tầng 1 + CBF Tầng 2 + Gemini LLM diễn giải).
- **Web Frontend:** Sử dụng **Streamlit** (tại `web/app.py`), kết nối qua HTTP/JSON tới API backend thông qua biến môi trường `API_URL`.

```
                    ┌────────────────────────┐
                    │  Gemini API (GenAI)    │
                    └───────────┬────────────┘
                                │ (Bước 4)
                                ▼
 ┌──────────────────────┐  HTTP/JSON  ┌──────────────────────────────────────────────┐
 │ Giao diện Web        │◄───────────►│             FASTAPI BACKEND                  │
 │ (Streamlit: web/)    │             │              (api/main.py)                   │
 └──────────────────────┘             │                                              │
                                      │  - POST /predict   (ML Inference Tầng 1)     │
                                      │  - POST /decision  (ML + CBF + Gemini LLM)   │
                                      │  - GET  /model-info (Dynamic Metadata)       │
                                      │  - GET  /health    (Service Health)          │
                                      └──────────────────────┬───────────────────────┘
                                                             │ nạp 1 lần
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │ models/current/              │
                                              │ pipeline + metadata          │
                                              └──────────────────────────────┘
```

---

### Luồng chạy ứng dụng

- **Luồng chạy duy nhất:** `api/main.py` (FastAPI) nạp model một lần và phục vụ suy luận; `web/app.py` (Streamlit) là giao diện người dùng duy nhất và gọi API.

---

## 3. Cấu trúc Thư mục

Cấu trúc dự án được chuẩn hóa theo đúng Mục 3.2 trong tài liệu hướng dẫn:

```
career_recommendation_v2/
├── career_ai/                    # Mã nguồn lõi: dữ liệu, model, train, đánh giá, giám sát
│   ├── domain_knowledge.py
│   ├── cbf_recommender.py
│   ├── decision_core.py
│   ├── model_exporter.py
│   ├── monitor_service.py
│   ├── feedback_store.py
│   ├── retrain_service.py
│   ├── prepare_data.py
│   ├── train.py
│   └── eval_system.py
├── data/
│   ├── DATASET.xlsx              # Dữ liệu nguồn
│   ├── students.csv              # Dữ liệu huấn luyện đã chuẩn bị (355 sinh viên)
│   ├── careers.json              # Danh mục 21 nghề thuộc 7 nhóm ngành CNTT
│   ├── feedback.json             # Phản hồi tường minh (Like/Dislike) từ người dùng (ban đầu rỗng)
│   └── interactions.json         # Nhật ký tương tác (ban đầu rỗng)
│                                 # (prediction_logs.jsonl được tạo tự động khi có lượt dự đoán, phục vụ giám sát drift)
├── models/
│   ├── current/                  # Pipeline triển khai và metadata đang dùng
│   └── v<phiên bản>/             # Các bản snapshot lịch sử
├── model/
│   └── 01_train_export_model.ipynb # Notebook EDA và huấn luyện
├── api/
│   ├── main.py                   # Bước 2: REST API phục vụ suy luận bằng FastAPI
│   ├── llm_explain.py            # Bước 4: Giải thích kết quả bằng Gemini API (google-genai SDK)
│   └── test_api.py               # Bước 5: Script kiểm thử tự động API (hợp lệ & không hợp lệ)
├── web/
│   └── app.py                    # Bước 3: Giao diện web tương tác bằng Streamlit
├── tests/
│   ├── test_system.py            # Kiểm thử hồi quy logic nghiệp vụ và độ nhất quán
│   ├── test_lifecycle.py         # Kiểm thử toàn diện 5 giai đoạn vòng đời mô hình
│   ├── test_llm_and_model.py     # Kiểm thử lớp LLM (trung thực, chịu lỗi), nhất quán model/metadata, bảo mật
│   └── test_consistency.py       # Kiểm tra API, đầu vào, log giám sát; chạy từ api/
├── .streamlit/
│   └── config.toml               # Cấu hình theme và server cho Streamlit
├── Dockerfile                    # Docker container hóa phục vụ triển khai cloud
├── render.yaml                   # Cấu hình triển khai tự động lên nền tảng Render
├── Procfile                      # File cấu hình tiến trình PaaS
├── start.sh                      # Kịch bản khởi chạy đồng thời API và Web
├── requirements.txt              # Danh sách thư viện phụ thuộc
├── .env.example                  # Mẫu biến môi trường (tự tạo .env từ file này; KHÔNG nộp/commit .env)
└── README.md                     # Tài liệu hướng dẫn sử dụng và triển khai
```

---

## 4. Hướng dẫn Cài đặt & Chuẩn bị Môi trường

### 4.1. Yêu cầu môi trường
- Python 3.10, 3.11 hoặc 3.14 (hỗ trợ Windows, Linux, macOS).
- Trình quản lý gói `pip`.

### 4.2. Cài đặt thư viện
Tại thư mục gốc của dự án, chạy:
```bash
pip install -r requirements.txt
```

### 4.3. Cấu hình biến môi trường
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```
Nội dung file `.env`:
```ini
GEMINI_API_KEY=            # để trống thì dùng giải thích theo quy tắc; lấy key tại https://aistudio.google.com/
GEMINI_MODEL=gemini-2.5-flash   # tên model có thể thay đổi, kiểm tra tài liệu hiện hành
GEMINI_TIMEOUT_MS=10000    # thời gian chờ tối đa khi gọi Gemini
API_URL=http://127.0.0.1:8000
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```
> **Lưu ý về Bảo mật:** Tuyệt đối không lưu khóa `GEMINI_API_KEY` trong code JavaScript client hay public repo. API key chỉ được backend đọc từ biến môi trường. Nếu chưa cấu hình key, hệ thống tự động kích hoạt bộ sinh giải thích quy tắc chuyên gia (Rule-based engine) mà không sập ứng dụng.

---

## 5. Trình Tự Thực Thi Từng Bước (Theo Tài Liệu Hướng Dẫn)

### Bước 1: Huấn luyện & Export Model
Huấn luyện 3 mô hình (`Logistic Regression`, `Random Forest`, `Gradient Boosting`), so sánh 5-fold CV và xuất mô hình tốt nhất:
```bash
python -m career_ai.prepare_data
python -m career_ai.train
```
*(Hoặc mở file Jupyter Notebook `model/01_train_export_model.ipynb` để theo dõi các biểu đồ EDA và ma trận nhầm lẫn).*

**Kết quả đánh giá trên tập dữ liệu:**
| Thuật toán | Accuracy | Precision | Recall | F1 (Test) | F1 (5-Fold CV) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (multinomial)** | **0.7887** | **0.7892** | **0.7887** | **0.7836** | **0.8539** | **<-- TRIỂN KHAI** |
| Random Forest | 0.7324 | 0.7370 | 0.7324 | 0.7240 | 0.6990 | |
| Gradient Boosting | 0.6761 | 0.6770 | 0.6761 | 0.6610 | 0.6840 | |

Pipeline tốt nhất và metadata được lưu trong `models/current/`; snapshot từng phiên bản nằm trong `models/v<phiên bản>/`. F1 5-fold CV = 0.854 ± 0.040 (độ lệch chuẩn được tính thật từ 5 fold). Số liệu có thể lệch nhẹ giữa các phiên bản scikit-learn; metadata ghi `sklearn_version`, và nếu chạy trên máy khác phiên bản, hãy chạy lại `python -m career_ai.train`.

> **Đây là độ khớp với nhãn SUY RA từ bảng điểm, không phải độ chính xác dự đoán nghề nghiệp thực tế** (xem Mục 7).

---

### Bước 2: Khởi động REST API Backend (FastAPI)
Chạy server API phục vụ tại cổng 8000:
```bash
uvicorn api.main:app --reload --port 8000
```
- Trang tài liệu tương tác Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Trang tài liệu ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Kiểm tra sức khỏe: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Cũng chạy được theo đúng cách trong PDF (từ trong thư mục `api/`): `cd api && uvicorn main:app --reload --port 8000` (API tự chuyển thư mục làm việc về thư mục gốc để đọc `data/`).
- Endpoint: `POST /predict` (chỉ mô hình học máy, không điều chỉnh), `POST /decision` (mô hình + sở thích/kỹ năng + CBF + giải thích), `GET /model-info`, `GET /health`. Đầu vào không hợp lệ bị từ chối với mã 422 (điểm âm/vượt 10, thiếu môn, NaN/Infinity, giá trị true/false, tên quá 100 ký tự, kỹ năng hoặc sở thích ngoài danh mục); thông báo lỗi không phản chiếu lại dữ liệu đã gửi.

---

### Bước 3: Khởi động Giao diện Web (Streamlit)
Mở cửa sổ dòng lệnh thứ hai và chạy:
```bash
streamlit run web/app.py
```
Ứng dụng web tự động mở tại: **[http://localhost:8501](http://localhost:8501)**
- Giao diện trực quan cho phép:
  - Chọn nhanh các hồ sơ mẫu sinh viên (AI/Data, Bảo mật, Web, Game).
  - Tải file bảng điểm Excel/CSV hoặc kéo thanh trượt điểm 12 môn học.
  - Chọn các mức độ kỹ năng mềm và lĩnh vực hứng thú.
  - Bấm nút **Phân Tích Định Hướng** để nhận báo cáo khuyến nghị toàn diện kèm biểu đồ và lời giải thích AI.

---

### Bước 4: Tích hợp Gemini LLM Diễn giải Kết quả
Module `api/llm_explain.py` dùng thư viện chính thức `google-genai`:
- Khi có `GEMINI_API_KEY`, `/decision` gọi Gemini để viết lời giải thích 3-5 câu. Có **timeout** (`GEMINI_TIMEOUT_MS`) và **cache** cho các yêu cầu giống nhau.
- Prompt chỉ chứa số liệu do hệ thống tính ra, nói rõ **tỷ lệ % chỉ là mức khớp giữa điểm các môn và hồ sơ môn học của nghề**, và cấm các cách nói khẳng định ("chắc chắn", "tối ưu", "bạn sẽ thành công"). Kết quả luôn kèm câu miễn trừ "gợi ý tham khảo".
- **Chống bịa số liệu bằng code:** mọi số thập phân/phần trăm trong câu trả lời của LLM phải nằm trong tập số liệu đã cung cấp; nếu không, câu trả lời bị loại và dùng bản theo quy tắc.
- Khi thiếu API key, hết hạn mức, mất mạng hoặc quá thời gian: tự dùng giải thích theo quy tắc (chỉ dùng dữ liệu có thật, không điền giá trị mặc định). Chi tiết lỗi chỉ ghi log phía server.
- Dữ liệu nhập tự do (sở thích, kỹ năng) được làm sạch trước khi đưa vào prompt; giao diện web escape HTML khi hiển thị nội dung LLM.
- **Quyền riêng tư:** thông tin tổng hợp từ kết quả (không kèm tên sinh viên) được gửi tới Google khi bật Gemini; giao diện có thông báo cho người dùng.

---

### Bước 5: Chạy Thử và Kiểm Tra Tự Động (API Testing)
Chạy kịch bản kiểm thử tự động tại `api/test_api.py`:
```bash
python api/test_api.py        # (khi API đang chạy)  hoặc: cd api && python test_api.py
python -m unittest discover -s tests   # bộ kiểm thử đơn vị/tích hợp (không cần bật API)
```
Kịch bản kiểm thử bao gồm các ca hợp lệ, các ca biên, và các ca dữ liệu lỗi (điểm âm, điểm vượt 10, thiếu trường, sai định dạng). Kết quả chạy thực tế:

```text
==============================================================================
BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG BỘ REST API (Bước 5 - PDF Mục 3.8)
==============================================================================
[OK ] Health check                             HTTP 200 (mong đợi 200) -   3 ms
[OK ] Lấy Model Info (Metadata)                HTTP 200 (mong đợi 200) -   3 ms
[OK ] SV thế mạnh Dữ liệu & AI                 HTTP 200 (mong đợi 200) -  33 ms
      -> Dự đoán: Dữ liệu & AI | Nghề #1: Chuyên viên phân tích dữ liệu / Nhà khoa học dữ liệu (98.5%)
[OK ] SV thế mạnh An toàn & Bảo mật            HTTP 200 (mong đợi 200) -  24 ms
      -> Dự đoán: An toàn thông tin | Nghề #1: Chuyên viên phân tích an ninh mạng (98.5%)
[OK ] SV thế mạnh Phát triển Web               HTTP 200 (mong đợi 200) -   6 ms
      -> Dự đoán: Phát triển Web | Nghề #1: Lập trình viên Frontend (98.5%)
[OK ] Điểm âm (ai = -2.0)                      HTTP 422 (mong đợi 422) -  23 ms
[OK ] Điểm vượt trần (db = 15.0)               HTTP 422 (mong đợi 422) -   3 ms
[OK ] Thiếu trường môn học (chỉ gửi 10 môn)    HTTP 422 (mong đợi 422) -   2 ms
[OK ] Sai kiểu dữ liệu (điểm là chuỗi chữ 'abc') HTTP 422 (mong đợi 422) -   2 ms
==============================================================================
[✓] TẤT CẢ CÁC CA KIỂM THỬ ĐỀU ĐẠT CHUẨN!
==============================================================================
```

Ngoài ra, chạy toàn bộ bộ kiểm thử hồi quy và vòng đời:
```bash
python -m unittest discover -s tests -v
```
*(Kết quả lần chạy gần nhất: 55/55 tests passed 100%).*

---

## 6. Hướng Dẫn Triển Khai Trực Tuyến (Online Deployment)

Để người dùng có thể truy cập qua trình duyệt mà không cần cài đặt code, dự án đã chuẩn bị sẵn đầy đủ file cấu hình cho các nền tảng đám mây phổ biến:

### Lựa chọn 1: Triển khai trên Streamlit Community Cloud (Nhanh nhất cho Web)
1. Đẩy mã nguồn dự án lên một kho GitHub (công khai hoặc riêng tư).
2. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
3. Nhấp **"New app"**, chọn kho lưu trữ và nhánh tương ứng.
4. Tại mục **Main file path**, điền: `web/app.py`.
5. Trong phần **Advanced settings -> Secrets**, thêm các biến:
   ```toml
   API_URL = "https://your-api-service.onrender.com"
   GEMINI_API_KEY = "your-gemini-key"
   ```
6. Bấm **Deploy**. Sau 2-3 phút, bạn sẽ có đường link trực tuyến công khai (dạng `https://career-ai.streamlit.app`).

### Lựa chọn 2: Triển khai toàn bộ trên Render (render.com)
Dự án đã có sẵn file `render.yaml` chuẩn hóa blueprint:
1. Đăng ký tài khoản tại [render.com](https://render.com).
2. Kết nối tài khoản GitHub của bạn với Render.
3. Tạo mới **Blueprint** và chọn repo này. Render sẽ tự động đọc `render.yaml` và tạo 2 dịch vụ:
   - `career-ai-api`: Chạy FastAPI backend (`uvicorn api.main:app --port $PORT`).
   - `career-ai-web`: Chạy Streamlit frontend (`streamlit run web/app.py`).
4. Nhập biến môi trường bí mật `GEMINI_API_KEY` trong Dashboard của Render.
5. Dịch vụ tự động build và cung cấp 2 đường link HTTPS hoạt động 24/7.

### Lựa chọn 3: Triển khai trên Hugging Face Spaces (Docker Space)
1. Tạo một Space mới trên [huggingface.co/spaces](https://huggingface.co/spaces), chọn SDK là **Docker**.
2. Đẩy toàn bộ source code (bao gồm `Dockerfile` và `start.sh`) lên repo Space.
3. Trong mục **Settings -> Variables and secrets**, thêm `GEMINI_API_KEY`.
4. Hugging Face sẽ tự động build image Docker và chạy cả backend lẫn frontend.

---

## 7. Giới Hạn và Ý Nghĩa Khoa Học

1. **Nhãn suy ra từ bảng điểm (Pseudo-labeling):** Vì `data/DATASET.xlsx` là bảng điểm thực tế của 355 sinh viên nhưng không ghi nghề nghiệp thực tế của từng bạn sau khi ra trường, nhãn `career_group` được xác định thông qua ma trận tri thức chuyên gia (`career_ai/domain_knowledge.py`). Chỉ số F1 5-fold CV phản ánh mức độ mô hình học lại quy tắc phân nhóm một cách ổn định, không phải độ chuẩn xác tuyệt đối ngoài đời thực.
2. **Phản hồi và huấn luyện lại:** Hai nút phản hồi trong Streamlit hiện chỉ hiển thị thông báo, chưa lưu phản hồi và chưa tự kích hoạt huấn luyện lại. `retrain_service.py` vẫn chứa logic huấn luyện lại từ dữ liệu feedback được chuẩn bị riêng.
3. **Vì sao vẫn dùng học máy khi nhãn là suy ra từ điểm?** Ở giai đoạn đầu, mô hình học xấp xỉ quy tắc gán nhãn (F1 5-fold CV = 0.854), nên giá trị của học máy chưa nằm ở độ chính xác vượt trội. Muốn đánh giá khả năng định hướng nghề thực tế cần thu thập nhãn nghề nghiệp được người dùng xác nhận và kiểm định độc lập.
4. **Sở thích và kỹ năng mềm không nằm trong dữ liệu huấn luyện.** Mô hình chỉ học từ 12 điểm môn; sở thích (+15% tương đối cho nhóm được chọn) và kỹ năng mềm (theo cấp độ) chỉ điều chỉnh xác suất bằng quy tắc sau khi mô hình đã dự đoán. Quy tắc này nằm ở `career_ai/decision_core.py`; `POST /predict` trả kết quả mô hình thuần túy, còn `/decision` trả cả xác suất gốc (`probabilities_model_only`) và xác suất sau điều chỉnh (`probabilities`). Kỹ năng mềm để "Chưa đánh giá" bị bỏ qua, không bị coi là "Trung bình".
5. **Tầng LLM chỉ diễn đạt lại kết quả,** không bổ sung thông tin về thị trường lao động hay năng lực thực tế của sinh viên.
6. **Bảo mật khi triển khai:** không nộp/commit file `.env`; chỉ cấu hình khóa Gemini ở backend và đặt `CORS_ORIGINS` đúng địa chỉ giao diện khi triển khai.
