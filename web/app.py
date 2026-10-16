# -*- coding: utf-8 -*-
"""
web/app.py — Bước 3: Giao diện Web bằng Streamlit (PDF Mục 3.5).

Nguyên tắc kiến trúc:
1. Giao diện KHÔNG nạp model trực tiếp; nó là một Web Client kết nối với FastAPI REST API qua HTTP.
2. Khi khởi động, giao diện gọi GET {API_URL}/model-info để lấy metadata và dựng form nhập liệu động.
3. Cho phép nhập tay điểm 12 môn (slider / number input 0.0 - 10.0) hoặc tải file Excel/CSV.
4. Gửi yêu cầu POST {API_URL}/decision và hiển thị kết quả theo định hướng Hệ hỗ trợ quyết định (DSS):
   - Nhóm ngành khuyến nghị & độ tin cậy.
   - Biểu đồ phân bố xác suất giữa 7 nhóm nghề IT.
   - Chi tiết Top 3 nghề nghiệp phù hợp nhất (độ phù hợp %, mức lương, xu hướng, môn trọng tâm).
   - Phân tích thế mạnh và các môn cần cải thiện.
   - Lời giải thích và tư vấn định hướng chuyên sâu từ Gemini AI (Bước 4).
   - Tương tác phản hồi người dùng.
5. Địa chỉ API đọc từ biến môi trường API_URL (mặc định: http://127.0.0.1:8000).

Cách chạy:
    streamlit run web/app.py
    hoặc: cd web && streamlit run app.py
"""
import html
import os
import sys
import io
import json
import time
import requests
import streamlit as st
import pandas as pd

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="AI Career DSS — Định Hướng Nghề Nghiệp CNTT",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
if not API_URL.startswith(("http://", "https://")):
    API_URL = "http://" + API_URL  # Render chỉ cấp host:port, không kèm scheme

try:
    import ui_kit as ui  # CSS + thành phần HTML/SVG của giao diện hiện đại (web/ui_kit.py)
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    import ui_kit as ui

st.markdown(ui.CSS, unsafe_allow_html=True)


def _normalize_text(value):
    """Chuẩn hoá chuỗi để so khớp tên môn học giữa Excel/CSV và SUBJECTS."""
    import unicodedata
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.replace("_", " ").replace("-", " ").replace("/", " ").split())


def parse_uploaded_scores(df, subject_names, subject_keys):
    """Trích xuất điểm 12 môn học từ file Excel/CSV theo nhiều định dạng khác nhau."""
    if df is None or df.empty:
        return {}

    def normalize_piece(v):
        return " ".join(
            _normalize_text(v)
            .replace("\xa0", " ")
            .replace("đ", "d")
            .replace("ơ", "o")
            .replace("ư", "u")
            .split()
        )

    course_code_map = {
        "th4306": "Công nghệ phần mềm",
        "th4320": "Trí tuệ nhân tạo",
        "th5206": "Mạng máy tính",
        "th5209": "Xử lý ảnh",
        "th4316": "Công nghệ Java",
        "th5208": "Phân tích thiết kế HTTT",
        "th5221": "Hệ quản trị CSDL",
        "th4315": "Ngôn ngữ C# và công nghệ .NET",
        "th5210": "An toàn và bảo mật HTTT",
        "th5231": "Kỹ thuật đồ hoạ máy tính",
        "th5211": "Hệ điều hành Linux",
        "th4309": "Công nghệ web",
    }

    alias_to_subject = {}
    for subj in subject_names:
        key = subject_keys.get(subj)
        aliases = [subj, key]
        if key:
            aliases.append(key.lower())
        for alias in aliases:
            norm = normalize_piece(alias)
            if norm:
                alias_to_subject.setdefault(norm, subj)

        subject_aliases = {
            "An toàn và bảo mật HTTT": ["an toan va bao mat htnn", "attt", "atbm", "bao mat", "security", "sec"],
            "Công nghệ phần mềm": ["cong nghe phan mem", "phan mem", "software engineering", "se"],
            "Hệ điều hành Linux": ["he dieu hanh linux", "linux"],
            "Hệ quản trị CSDL": ["he quan tri csdl", "csdl", "sql", "db"],
            "Kỹ thuật đồ hoạ máy tính": ["ky thuat do hoa may tinh", "do hoa", "graphics"],
            "Mạng máy tính": ["mang may tinh", "mmt", "network"],
            "Phân tích thiết kế HTTT": ["phan tich thiet ke htnn", "analysis"],
            "Xử lý ảnh": ["xu ly anh", "image processing", "image"],
            "Công nghệ Java": ["cong nghe java", "java"],
            "Ngôn ngữ C# và công nghệ .NET": ["ngon ngu csharp", "csharp", "dotnet"],
            "Công nghệ web": ["cong nghe web", "web"],
            "Trí tuệ nhân tạo": ["tri tue nhan tao", "ai"],
        }
        for alias_name, extra_aliases in subject_aliases.items():
            if normalize_piece(alias_name) == normalize_piece(subj):
                for extra in extra_aliases:
                    alias_to_subject.setdefault(normalize_piece(extra), subj)
        for code, subject_name in course_code_map.items():
            if normalize_piece(subject_name) == normalize_piece(subj):
                alias_to_subject.setdefault(code.lower(), subj)

    def match_subject_name(label):
        norm = normalize_piece(label)
        if not norm:
            return None
        if norm in alias_to_subject:
            return alias_to_subject[norm]

        compact = "".join(char for char in norm if char.isalnum())
        for code, subject_name in course_code_map.items():
            if code in compact:
                return subject_name

        candidates = []
        for alias, subject in alias_to_subject.items():
            if len(alias) > 2 and alias in norm:
                candidates.append((len(alias), subject))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        return None

    def parse_score(raw_value):
        if raw_value is None or pd.isna(raw_value):
            return None
        try:
            value = float(str(raw_value).replace(",", ".").strip())
        except (TypeError, ValueError):
            return None
        return value if 0.0 <= value <= 10.0 else None

    def assign_score(subject_name, raw_value):
        val = parse_score(raw_value)
        if val is not None:
            key = subject_keys.get(subject_name)
            if key:
                scores[key] = val
                return True
        return False

    scores = {}

    # 1) Dạng ngang: cột tên môn hoặc mã môn là tên cột; lấy điểm từ cột đó.
    for subj in subject_names:
        if subject_keys.get(subj) in scores:
            continue
        for col in df.columns:
            if match_subject_name(col) == subj:
                for value in df[col].tolist():
                    if assign_score(subj, value):
                        break
                if subject_keys.get(subj) in scores:
                    break

    if len(scores) < len(subject_names):
        norm_cols = [normalize_piece(c) for c in df.columns]
        name_candidates = [
            i for i, c in enumerate(norm_cols)
            if any(token in c for token in [
                "ten mon", "ten hoc phan", "mon hoc", "hoc phan", "ma mon", "ma hoc phan",
                "ky hieu", "subject", "course", "course name", "subject name", "course code"
            ])
        ]
        score_candidates = [
            i for i, c in enumerate(norm_cols)
            if any(token in c for token in [
                "tbchp", "tbhp", "diem", "tong ket", "diem tb", "score", "marks",
                "grade", "point", "diem so", "final"
            ])
        ]
        if name_candidates and score_candidates:
            preferred_scores = sorted(
                score_candidates,
                key=lambda i: (
                    not any(token in norm_cols[i] for token in ("tbchp", "tbhp", "tong ket", "diem tb", "final")),
                    i,
                ),
            )
            best_scores = {}
            for name_idx in name_candidates:
                for score_idx in preferred_scores:
                    candidate_scores = {}
                    for _, row in df.iterrows():
                        matched = match_subject_name(row.iloc[name_idx])
                        value = parse_score(row.iloc[score_idx])
                        key = subject_keys.get(matched) if matched is not None else None
                        if key and value is not None:
                            candidate_scores[key] = value
                    if len(candidate_scores) > len(best_scores):
                        best_scores = candidate_scores
            scores.update(best_scores)

    if len(scores) < len(subject_names):
        for subj in subject_names:
            if subject_keys.get(subj) in scores:
                continue
            for _, row in df.iterrows():
                for col_name, raw_value in row.items():
                    if match_subject_name(col_name) == subj:
                        assign_score(subj, raw_value)
                    elif match_subject_name(raw_value) == subj:
                        assign_score(subj, raw_value)
                    if subject_keys.get(subj) in scores:
                        break
                if subject_keys.get(subj) in scores:
                    break

    return scores


def parse_uploaded_file(uploaded_file, filename, subject_names, subject_keys):
    """Parse CSV/XLSX sheets and try header rows near the top of each sheet."""
    if filename.lower().endswith(".csv"):
        sheets = {"CSV": pd.read_csv(uploaded_file, header=None)}
    else:
        sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None)

    best_scores = {}
    max_columns = 0
    for raw_df in sheets.values():
        max_columns = max(max_columns, raw_df.shape[1])
        for header_row in range(min(len(raw_df), 12)):
            headers = raw_df.iloc[header_row].fillna("").astype(str).str.strip().tolist()
            if not any(headers):
                continue
            body = raw_df.iloc[header_row + 1:].copy()
            body.columns = headers
            parsed = parse_uploaded_scores(body, subject_names, subject_keys)
            if len(parsed) > len(best_scores):
                best_scores = parsed
            if len(best_scores) == len(subject_names):
                return best_scores, max_columns
    return best_scores, max_columns


# ---------------------------------------------------------------------------
# HÀM LẤY METADATA TỪ API BACKEND
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_model_info():
    """Lấy thông tin mô hình và danh mục từ endpoint /model-info."""
    try:
        res = requests.get(f"{API_URL}/model-info", timeout=4)
        if res.status_code == 200:
            return res.json(), None
        return None, f"API trả về mã lỗi HTTP {res.status_code}"
    except requests.exceptions.RequestException as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# GIAO DIỆN CHÍNH
# ---------------------------------------------------------------------------
def main():
    st.markdown(ui.hero(), unsafe_allow_html=True)
    meta, error = fetch_model_info()

    with st.sidebar:
        st.markdown("#### Trạng thái hệ thống")
        if meta:
            m = meta.get("metrics_test", {}) or {}
            st.markdown(ui.status_card(True, meta.get("algorithm", "Logistic Regression"), meta.get("version", "1.0.0"),
                                       m.get("f1_cv_5fold"), API_URL), unsafe_allow_html=True)
            if meta.get("metrics_note"):
                st.caption(meta["metrics_note"])
        else:
            st.markdown(ui.status_card(False, "-", "-", None, API_URL), unsafe_allow_html=True)
            st.error(f"Không thể kết nối tới Backend API: {error}")
            st.code("uvicorn api.main:app --reload --port 8000", language="bash")
            st.stop()
        with st.expander("Quyền riêng tư"):
            st.write("Điểm, kỹ năng và sở thích bạn nhập được gửi tới máy chủ API. Nếu bật Gemini, thông tin tổng hợp "
                     "từ kết quả (không kèm tên) cũng được gửi tới Google. Chỉ nhập dữ liệu bạn đồng ý chia sẻ.")
        with st.expander("Hệ thống hoạt động thế nào?"):
            st.write("**Tầng 1 (ML):** phân loại xác suất 7 nhóm ngành từ 12 điểm môn.  \n"
                     "**Tầng 2 (CBF):** xếp hạng 21 nghề theo môn cốt lõi và kỹ năng mềm.  \n"
                     "**Tầng 3 (Gemini):** diễn giải kết quả bằng ngôn ngữ tự nhiên.")

    st.write("")
    with st.container(border=True):
        st.markdown(ui.section(1, "Bảng điểm 12 học phần"), unsafe_allow_html=True)
        st.markdown(
            "<div class='input-hint'><span class='dot'></span>"
            "Nhập điểm trực tiếp hoặc tải bảng điểm để tự động điền · Thang điểm 0–10</div>",
            unsafe_allow_html=True,
        )

        tab_manual, tab_file = st.tabs(["✎  Nhập điểm thủ công", "⇧  Tải bảng điểm"])

        input_scores = {}
        subject_names = meta.get("subject_names", [])
        subject_keys = meta.get("subject_keys", {})

        with tab_manual:
            st.caption("Nhập điểm từng học phần theo thang 10. Có thể dùng nút tăng/giảm hoặc gõ trực tiếp.")
            cols = st.columns(3)

            for i, subj in enumerate(subject_names):
                k = subject_keys.get(subj, f"s_{i}")
                col = cols[i % 3]
                default_val = 0.0
                input_scores[k] = col.number_input(
                    f"{subj}",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(default_val),
                    step=0.1,
                    format="%.1f",
                    key=f"score_input_{k}",
                )

        with tab_file:
            uploaded_file = st.file_uploader(
                "Chọn bảng điểm cá nhân",
                type=["xlsx", "csv"],
                help="Hỗ trợ Excel (.xlsx) và CSV (.csv). Hệ thống sẽ tự tìm tên học phần và điểm tổng kết.",
            )
            if uploaded_file is not None:
                try:
                    parsed_scores, column_count = parse_uploaded_file(
                        uploaded_file, uploaded_file.name, subject_names, subject_keys
                    )
                    st.success(
                        f"Đã đọc file '{uploaded_file.name}' thành công! "
                        f"Tìm thấy tối đa {column_count} cột dữ liệu."
                    )
                    found_count = len(parsed_scores)
                    if found_count >= 12:
                        input_scores.update(parsed_scores)
                        st.info(f"Đã tự động trích xuất điểm của {found_count}/12 môn học từ file.")
                    else:
                        st.warning(
                            f"File được tải nhưng chỉ tìm thấy {found_count}/12 môn học. "
                            "Hãy kiểm tra sheet có bảng điểm, tên/mã học phần và cột điểm tổng kết; "
                            "hoặc nhập tay điểm còn thiếu."
                        )
                except Exception as e:
                    st.error(f"Lỗi khi đọc file: {str(e)}")

    with st.container(border=True):
        # 3. Kỹ năng mềm & Sở thích
        st.markdown(ui.section(2, "Kỹ năng mềm & lĩnh vực quan tâm"), unsafe_allow_html=True)
        col_skills, col_interests = st.columns(2)

        soft_skills_cfg = meta.get("soft_skills", [])
        skill_levels_cfg = meta.get("skill_levels", [])
        # Thêm lựa chọn "Chưa đánh giá" (mặc định). Kỹ năng chưa đánh giá không được gửi lên API nên không cộng/trừ điểm.
        UNASSESSED_LABEL = "Chưa đánh giá"
        level_labels = [UNASSESSED_LABEL] + [opt["label"] for opt in skill_levels_cfg]
        level_keys = [""] + [opt["key"] for opt in skill_levels_cfg]

        user_skills = {}
        with col_skills:
            st.markdown("**Đánh giá mức độ thành thạo Kỹ năng mềm:**")
            cols_s = st.columns(2)
            for idx, sk in enumerate(soft_skills_cfg):
                sk_key = sk["key"]
                sk_label = sk["label"]
                col_s = cols_s[idx % 2]
                chosen_label = col_s.selectbox(
                    sk_label,
                    options=level_labels,
                    index=0,
                    key=f"skill_select_{sk_key}",
                )
                chosen_key = level_keys[level_labels.index(chosen_label)]
                if chosen_key:
                    user_skills[sk_key] = chosen_key

        with col_interests:
            st.markdown("**Lĩnh vực bạn cảm thấy hứng thú nhất:**")
            interest_options = meta.get("interest_options", [])
            user_interests = st.multiselect(
                "Chọn một hoặc nhiều lĩnh vực quan tâm:",
                options=interest_options,
                default=[],
                key="interests_multiselect",
            )


    st.write("")
    submit_button = st.button("Phân tích định hướng nghề nghiệp", type="primary", use_container_width=True)

    if submit_button:
        payload = {"scores": input_scores, "skills": user_skills, "interests": user_interests, "student_name": "Sinh viên"}
        with st.spinner("Hệ thống AI đang phân tích hồ sơ của bạn..."):
            try:
                t0 = time.time()
                res = requests.post(f"{API_URL}/decision", json=payload, timeout=20)
                ms = round((time.time() - t0) * 1000)
                if res.status_code == 200:
                    st.session_state["result"] = {"data": res.json(), "scores": dict(input_scores), "ms": ms}
                    st.session_state.pop("error", None)
                elif res.status_code == 422:
                    st.session_state["error"] = f"Dữ liệu đầu vào không hợp lệ (HTTP 422): {res.json().get('detail')}"
                else:
                    st.session_state["error"] = f"Lỗi từ máy chủ API (HTTP {res.status_code}): {res.text[:300]}"
            except requests.exceptions.RequestException as ex:
                st.session_state["error"] = f"Không thể kết nối đến API tại {API_URL}: {ex}"

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    elif st.session_state.get("result"):
        render_results(st.session_state["result"], meta)


def render_results(r: dict, meta: dict):
    """Hiển thị kết quả từ session_state nên không biến mất khi người dùng bấm nút phản hồi."""
    d, scores = r["data"], r["scores"]
    top = d.get("top_careers", []) or []
    llm = d.get("explanation_llm", {}) or {}
    st.markdown("<div id='ket-qua'></div>", unsafe_allow_html=True)
    st.markdown(ui.result_hero(d.get("predicted_group", ""), float(d.get("confidence", 0) or 0),
                               top[0].get("career_name", "") if top else "", float(top[0].get("match_percentage", 0)) if top else 0.0,
                               bool(d.get("adjusted_by_preferences"))), unsafe_allow_html=True)
    st.caption(f"Phân tích trong {r['ms']} ms · mô hình v{meta.get('version', '?')}")

    c1, c2 = st.columns(2, gap="medium")
    c1.markdown(ui.radar_svg(scores), unsafe_allow_html=True)
    c2.markdown(ui.prob_bars(d.get("probabilities", {})), unsafe_allow_html=True)

    st.markdown(ui.section(3, "Giải thích kết quả"), unsafe_allow_html=True)
    st.markdown(ui.explain_card(llm.get("explanation", ""), llm.get("note", ""), bool(llm.get("is_ai_generated")),
                                llm.get("model_used", "Gemini"), d.get("disclaimer", llm.get("disclaimer", ""))), unsafe_allow_html=True)

    st.markdown(ui.section(4, "Top 3 nghề khớp nhất với điểm của bạn"), unsafe_allow_html=True)
    st.markdown(ui.career_grid(top), unsafe_allow_html=True)  # lưới CSS: 3 thẻ luôn cao bằng nhau

    st.markdown(ui.section(5, "Đối sánh học phần cá nhân"), unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    a.markdown(ui.tags(d.get("strengths", []), "good", "Điểm các môn khá đồng đều.", "Thế mạnh nổi trội"), unsafe_allow_html=True)
    b.markdown(ui.tags(d.get("improvements", []), "todo", "Các môn cơ sở đều đạt tốt.", "Cần trau dồi thêm"), unsafe_allow_html=True)

    st.write("")
    st.markdown("**Kết quả này có hữu ích với bạn không?**")
    f1, f2, _ = st.columns([1.2, 1.5, 2])
    if f1.button("Phù hợp với tôi", key="fb_like"):
        st.toast("Cảm ơn bạn! Phản hồi hiện chưa được lưu.")
    if f2.button("Chưa đúng nguyện vọng", key="fb_dislike"):
        st.toast("Cảm ơn bạn! Phản hồi hiện chưa được lưu.")
    st.caption("Phản hồi hiện chỉ hiển thị thông báo, chưa được lưu vào dữ liệu huấn luyện.")


if __name__ == "__main__":
    main()
