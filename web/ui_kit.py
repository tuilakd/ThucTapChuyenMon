# -*- coding: utf-8 -*-
"""
web/ui_kit.py — Bộ thành phần giao diện hiện đại cho web/app.py (CSS + hàm dựng HTML/SVG).
Không thêm thư viện: radar, vòng tin cậy, thanh xác suất đều là HTML/SVG thuần.
Mọi chuỗi động đều đi qua html.escape() trước khi chèn vào HTML.
"""
import html
import math

PALETTE = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#64748B"]
AXIS_LABEL = {
    "sec": "ATBM", "se": "CNPM", "linux": "Linux", "db": "CSDL", "graphics": "Đồ họa", "network": "Mạng",
    "analysis": "PTTK", "image": "Xử lý ảnh", "java": "Java", "csharp": "C#/.NET", "web": "Web", "ai": "AI",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#0F172A;--muted:#64748B;--line:#E2E8F0;--card:#FFFFFF;--bg:#F5F7FB;--brand:#4F46E5;--brand2:#0EA5E9;--ok:#059669;--warn:#D97706}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(1200px 500px at 85% -10%,#E0E7FF 0%,transparent 60%),var(--bg)}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding-top:1.6rem;max-width:1180px}
h1,h2,h3,h4{letter-spacing:-.01em}

/* Hero */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:2rem 2.2rem;color:#fff;
 background:linear-gradient(120deg,#312E81 0%,#4F46E5 48%,#0EA5E9 100%);box-shadow:0 20px 40px -18px rgba(79,70,229,.55)}
.hero:after{content:"";position:absolute;right:-60px;top:-80px;width:260px;height:260px;border-radius:50%;background:rgba(255,255,255,.10)}
.hero h1{margin:0 0 .35rem;font-size:2rem;font-weight:800;color:#fff;line-height:1.2}
.hero p{margin:0;opacity:.88;font-size:1.02rem;max-width:760px}
.pills{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.1rem}
.pill{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);padding:.32rem .8rem;border-radius:999px;font-size:.82rem;font-weight:600;backdrop-filter:blur(4px)}

/* Section / card */
.sec-title{display:flex;align-items:center;gap:.7rem;font-weight:700;font-size:1.12rem;margin:.1rem 0 .7rem}
.step{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:9px;background:var(--brand);color:#fff;font-size:.85rem;font-weight:700}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid var(--line)!important;border-radius:18px!important;box-shadow:0 8px 24px -16px rgba(15,23,42,.25)}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.2rem 1.3rem;box-shadow:0 8px 24px -16px rgba(15,23,42,.25);height:100%}
.card h4{margin:0 0 .8rem;font-size:1rem;font-weight:700}

/* Inputs */
.stButton>button[kind="primary"]{background:linear-gradient(120deg,#4F46E5,#0EA5E9);border:0;border-radius:14px;padding:.8rem 1rem;font-weight:700;font-size:1.05rem;box-shadow:0 12px 24px -12px rgba(79,70,229,.7);transition:transform .15s}
.stButton>button[kind="primary"]:hover{transform:translateY(-1px)}
.stButton>button[kind="secondary"]{border-radius:12px;border:1px solid var(--line);font-weight:600}
[data-baseweb="select"]>div,[data-baseweb="tab-list"]{border-radius:12px}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}

/* Grade input */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid #E2E8F0;padding:0 .15rem}
[data-testid="stTabs"] button[data-baseweb="tab"]{height:44px;padding:0 1rem;border-radius:11px 11px 0 0;color:#64748B;font-weight:600}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]{color:#4338CA;background:#EEF2FF}
[data-testid="stFileUploader"] section{border:1.5px dashed #C7D2FE;border-radius:16px;background:linear-gradient(135deg,#F8FAFF,#F5F7FF);padding:1rem;transition:border-color .18s,background .18s}
[data-testid="stFileUploader"] section:hover{border-color:#818CF8;background:#EEF2FF}
[data-testid="stFileUploader"] button{border-radius:10px!important;border:1px solid #C7D2FE!important;color:#4338CA!important;font-weight:600!important}
[data-testid="stFileUploader"] small{color:#64748B}
[data-testid="stNumberInput"] input{border-radius:10px;font-weight:600}
[data-testid="stAlert"]{border-radius:13px}
.input-hint{display:flex;align-items:center;gap:.55rem;margin:.15rem 0 1rem;color:#64748B;font-size:.9rem}
.input-hint .dot{width:8px;height:8px;background:#10B981;border-radius:50%;box-shadow:0 0 0 4px #D1FAE5;flex:none}

/* Result hero */
.res{display:flex;gap:1.5rem;align-items:center;justify-content:space-between;border-radius:22px;padding:1.6rem 2rem;color:#fff;
 background:linear-gradient(125deg,#0F172A 0%,#312E81 60%,#4F46E5 100%);box-shadow:0 22px 44px -22px rgba(49,46,129,.7);margin:1rem 0}
.res .k{font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;opacity:.7;font-weight:600}
.res .v{font-size:2.1rem;font-weight:800;line-height:1.15;margin:.25rem 0 .5rem}
.res .s{opacity:.9;font-size:1rem}
.chip{display:inline-block;margin:.5rem .4rem 0 0;padding:.25rem .7rem;border-radius:999px;font-size:.78rem;font-weight:600;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25)}
.chip.warn{background:#FEF3C7;color:#92400E;border-color:#FCD34D}
.ring{--p:0;flex:0 0 auto;width:132px;height:132px;border-radius:50%;display:grid;place-items:center;
 background:conic-gradient(#34D399 calc(var(--p)*1%),rgba(255,255,255,.18) 0)}
.ring>div{width:104px;height:104px;border-radius:50%;background:#1E1B4B;display:grid;place-items:center;text-align:center}
.ring b{font-size:1.35rem;font-weight:800;display:block;line-height:1}.ring small{font-size:.62rem;opacity:.7;letter-spacing:.08em;text-transform:uppercase}

/* Probability bars */
.bar{margin:.55rem 0}.bar .t{display:flex;justify-content:space-between;font-size:.88rem;font-weight:600;margin-bottom:.25rem}
.bar .tr{height:10px;border-radius:999px;background:#EEF2F7;overflow:hidden}.bar .fl{height:100%;border-radius:999px}

/* Career cards */
.career{position:relative;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.2rem;box-shadow:0 8px 24px -16px rgba(15,23,42,.25);transition:transform .2s,box-shadow .2s}
.career:hover{transform:translateY(-3px);box-shadow:0 18px 32px -18px rgba(79,70,229,.45)}
.career.first{border:1.5px solid #A5B4FC;background:linear-gradient(180deg,#fff,#F5F3FF)}
.rank{display:inline-flex;align-items:center;gap:.4rem;font-size:.78rem;font-weight:700;color:#4338CA;background:#E0E7FF;padding:.25rem .65rem;border-radius:999px}
.career h3{font-size:1.08rem;font-weight:700;margin:.7rem 0 .4rem;line-height:1.3}
.career p{font-size:.86rem;color:#475569;line-height:1.5;min-height:62px;margin:0 0 .7rem}
.meter{display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem}.meter .tr{flex:1;height:8px;background:#EEF2F7;border-radius:999px;overflow:hidden}
.meter .fl{height:100%;background:linear-gradient(90deg,#4F46E5,#0EA5E9);border-radius:999px}.meter b{font-size:.88rem;color:#4338CA}
.kv{display:flex;justify-content:space-between;gap:.6rem;font-size:.82rem;padding:.4rem 0;border-top:1px dashed var(--line)}.kv span:first-child{color:var(--muted);white-space:nowrap}.kv span:last-child{text-align:right;font-weight:600}.kv.stack{flex-direction:column;gap:.15rem}.kv.stack span:last-child{text-align:left;font-weight:500;color:#334155;line-height:1.45}
.cgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;align-items:stretch}

/* Explanation + chips */
.explain{border-radius:18px;padding:1.3rem 1.5rem;background:linear-gradient(135deg,#EEF2FF,#F0F9FF);border:1px solid #C7D2FE}
.explain .b{display:inline-block;font-size:.78rem;font-weight:700;color:#3730A3;background:#fff;border:1px solid #C7D2FE;padding:.2rem .65rem;border-radius:999px;margin-bottom:.6rem}
.explain .x{font-size:1.02rem;line-height:1.7;color:#1E293B}.explain .n{margin-top:.6rem;font-size:.78rem;color:var(--muted)}
.tag{display:inline-block;margin:.25rem .3rem .25rem 0;padding:.35rem .8rem;border-radius:999px;font-size:.84rem;font-weight:600}
.tag.good{background:#D1FAE5;color:#065F46}.tag.todo{background:#FEF3C7;color:#92400E}
.note{font-size:.82rem;color:var(--muted);margin:.6rem 0 0}
@media(max-width:760px){.res{flex-direction:column;align-items:flex-start}.hero h1{font-size:1.5rem}}
</style>
"""


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def hero() -> str:
    pills = "".join(f"<span class='pill'>{e(t)}</span>" for t in
                    ["Tầng 1 · Học máy", "Tầng 2 · Lọc theo nội dung (CBF)", "Tầng 3 · Gemini giải thích"])
    return ("<div class='hero'><h1>Định hướng nghề nghiệp CNTT bằng AI</h1>"
            "<p>Nhập điểm 12 học phần, nhận gợi ý nhóm ngành, Top 3 nghề phù hợp và lời giải thích dễ hiểu. "
            "Kết quả mang tính tham khảo, giúp bạn ra quyết định tốt hơn.</p>"
            f"<div class='pills'>{pills}</div></div>")


def section(n: int, title: str) -> str:
    return f"<div class='sec-title'><span class='step'>{n}</span>{e(title)}</div>"


def status_card(ok: bool, algo: str, version: str, f1_cv, api_url: str) -> str:
    dot = "#10B981" if ok else "#EF4444"
    txt = "API đang hoạt động" if ok else "Mất kết nối API"
    f1 = f"{f1_cv:.3f}" if isinstance(f1_cv, (int, float)) else "n/a"
    return (f"<div style='font-weight:700;display:flex;align-items:center;gap:.5rem'>"
            f"<span style='width:10px;height:10px;border-radius:50%;background:{dot};box-shadow:0 0 0 4px {dot}33'></span>{txt}</div>"
            f"<div class='kv'><span>Thuật toán</span><span>{e(algo)}</span></div>"
            f"<div class='kv'><span>Phiên bản</span><span>v{e(version)}</span></div>"
            f"<div class='kv'><span>F1 5-fold CV</span><span>{f1}</span></div>"
            f"<div class='kv'><span>Backend</span><span style='font-size:.72rem'>{e(api_url)}</span></div>")


def result_hero(group: str, conf: float, top_name: str, top_pct: float, adjusted: bool) -> str:
    chips = ""
    if adjusted:
        chips += "<span class='chip'>Đã điều chỉnh theo sở thích / kỹ năng mềm</span>"
    if conf < 50:
        chips += "<span class='chip warn'>Độ tin cậy thấp: hồ sơ chưa nghiêng rõ về nhóm nào</span>"
    sub = f"Nghề khớp nhất: <b>{e(top_name)}</b> ({top_pct:.1f}% khớp hồ sơ môn học)" if top_name else ""
    return (f"<div class='res'><div><div class='k'>Nhóm ngành phù hợp nhất</div><div class='v'>{e(group)}</div>"
            f"<div class='s'>{sub}</div>{chips}</div>"
            f"<div class='ring' style='--p:{max(0, min(100, conf)):.0f}'><div><span><b>{conf:.1f}%</b><small>tin cậy</small></span></div></div></div>")


def prob_bars(probs: dict) -> str:
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    out = ["<div class='card'><h4>Phân bố xác suất 7 nhóm nghề</h4>"]
    for i, (g, p) in enumerate(items):
        out.append(f"<div class='bar'><div class='t'><span>{e(g)}</span><span>{p:.1f}%</span></div>"
                   f"<div class='tr'><div class='fl' style='width:{max(p, 0.6):.1f}%;background:{PALETTE[i % len(PALETTE)]}'></div></div></div>")
    out.append("<div class='note'>Xác suất do mô hình học máy tính từ 12 điểm môn học.</div></div>")
    return "".join(out)


def radar_svg(scores: dict) -> str:
    keys = [k for k in AXIS_LABEL if k in scores]
    n, cx, cy, R = len(keys), 210, 190, 128
    def pt(i, r):
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * math.cos(a), cy + r * math.sin(a)
    g = []
    for lv in (2.5, 5, 7.5, 10):
        g.append("<polygon points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, R * lv / 10) for i in range(n)))
                 + "' fill='none' stroke='#E2E8F0'/>")
    for i in range(n):
        x, y = pt(i, R)
        g.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.1f}' y2='{y:.1f}' stroke='#E2E8F0'/>")
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, R * max(0, min(10, float(scores[k]))) / 10) for i, k in enumerate(keys)))
    g.append(f"<polygon points='{poly}' fill='rgba(79,70,229,.22)' stroke='#4F46E5' stroke-width='2.5'/>")
    for i, k in enumerate(keys):
        x, y = pt(i, R * max(0, min(10, float(scores[k]))) / 10)
        g.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#4F46E5'/>")
        lx, ly = pt(i, R + 26)
        anc = "middle" if abs(lx - cx) < 12 else ("start" if lx > cx else "end")
        g.append(f"<text x='{lx:.1f}' y='{ly:.1f}' text-anchor='{anc}' dominant-baseline='middle' font-size='12' font-weight='600' fill='#334155'>"
                 f"{e(AXIS_LABEL[k])} <tspan fill='#4F46E5'>{float(scores[k]):.1f}</tspan></text>")
    return ("<div class='card'><h4>Hồ sơ năng lực 12 môn</h4>"
            f"<svg viewBox='-40 0 500 380' width='100%' role='img' aria-label='Biểu đồ radar điểm 12 môn'>{''.join(g)}</svg></div>")


def explain_card(text: str, note: str, is_ai: bool, model: str, disclaimer: str) -> str:
    badge = f"Giải thích do {e(model or 'Gemini')} viết" if is_ai else "Giải thích theo quy tắc"
    return (f"<div class='explain'><span class='b'>{badge}</span><div class='x'>{e(text)}</div>"
            f"<div class='n'>{e(note)}</div></div><div class='note'>{e(disclaimer)}</div>")


def career_card(idx: int, c: dict) -> str:
    pct = float(c.get("match_percentage", 0) or 0)
    medal = ["1", "2", "3"][idx] if idx < 3 else str(idx + 1)
    return (f"<div class='career{' first' if idx == 0 else ''}'><span class='rank'>Hạng {medal}</span>"
            f"<h3>{e(c.get('career_name'))}</h3><p>{e(c.get('description', ''))}</p>"
            f"<div class='meter'><div class='tr'><div class='fl' style='width:{max(0, min(100, pct)):.1f}%'></div></div><b>{pct:.1f}%</b></div>"
            f"<div class='kv'><span>Mức lương</span><span>{e(c.get('salary', 'Thỏa thuận'))}</span></div>"
            f"<div class='kv stack'><span>Xu hướng</span><span>{e(c.get('trend', 'Phát triển'))}</span></div></div>")


def career_grid(careers: list) -> str:
    return "<div class='cgrid'>" + "".join(career_card(i, c) for i, c in enumerate(careers[:3])) + "</div>"


def tags(items, kind: str, empty: str, title: str) -> str:
    body = "".join(f"<span class='tag {kind}'>{e(i)}</span>" for i in items) or f"<div class='note'>{e(empty)}</div>"
    return f"<div class='card'><h4>{e(title)}</h4>{body}</div>"
