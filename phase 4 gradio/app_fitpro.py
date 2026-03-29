"""
Phase 4: 헬스 기구 인식 Gradio 웹 서비스
- Stitch 테마 (디즈니 스티치 컬러 팔레트)
- 3페이지 구조: 프로필 → 기구 인식 → 운동 가이드
- iPhone 앱 대비 모바일 퍼스트 디자인
"""

import gradio as gr
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import base64
import mimetypes

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "gym_yolo26s_v1" / "weights" / "best.pt"
JSON_PATH  = BASE_DIR / "equipment_guide_v4.json"
IMAGE_DIR  = (BASE_DIR.parent / "images").resolve()   # gym/data/images

# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
with open(JSON_PATH, "r", encoding="utf-8") as f:
    GUIDE_DATA = json.load(f)

EQUIPMENT  = GUIDE_DATA["equipment_list"]
HR_ZONES   = GUIDE_DATA["heart_rate_zones"]

# YOLO 모델 로드
model = YOLO(str(MODEL_PATH))

# 클래스 인덱스 → 영문 이름 매핑
CLASS_NAMES = model.names   # {0: 'Ab_Wheel', 1: 'Aerobic_Stepper', ...}

# 영문→한글 매핑 (JSON 기준)
EN_TO_KO = {}
for eng_key, info in EQUIPMENT.items():
    EN_TO_KO[eng_key] = info.get("ko_name", eng_key)

# ═══════════════════════════════════════════
# Stitch 테마 CSS
# ═══════════════════════════════════════════
STITCH_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── CSS Variables (FitPro Dark Theme) ── */
:root {
    --fp-accent: #4ECDC4;
    --fp-accent-dim: rgba(78, 205, 196, 0.15);
    --fp-accent-glow: rgba(78, 205, 196, 0.25);
    --fp-green: #2ECC71;
    --fp-green-dim: rgba(46, 204, 113, 0.12);
    --fp-orange: #F39C12;
    --fp-red: #E74C3C;
    --fp-red-dim: rgba(231, 76, 60, 0.10);
    --fp-blue: #3498DB;

    --fp-bg: #000000;
    --fp-surface: #0D0D0D;
    --fp-card: #1A1A1A;
    --fp-card-alt: #141414;
    --fp-border: rgba(255, 255, 255, 0.06);
    --fp-border-accent: rgba(78, 205, 196, 0.18);

    --fp-text: #F0F0F0;
    --fp-text-secondary: #999999;
    --fp-text-muted: #666666;

    --radius: 14px;
    --radius-sm: 10px;
    --radius-lg: 20px;
    --radius-pill: 100px;
    --transition: all 0.25s ease;
}

/* ── Global ── */
* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Noto Sans KR', 'Inter', -apple-system, sans-serif !important;
    background: var(--fp-bg) !important;
    color: var(--fp-text) !important;
    -webkit-font-smoothing: antialiased;
}
.gradio-container {
    max-width: 480px !important;
    margin: 0 auto !important;
    min-height: 100vh;
}

/* ── Header ── */
.stitch-header {
    text-align: center;
    padding: 32px 16px 16px;
}
.stitch-header h1 {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--fp-text);
    letter-spacing: -0.5px;
}
.stitch-header p {
    color: var(--fp-text-secondary);
    font-size: 13px;
    margin-top: 4px;
    font-weight: 400;
}
.stitch-emoji {
    font-size: 44px;
    display: block;
    margin-bottom: 10px;
}

/* ── Tabs (hidden navigation) ── */
.tabs { border: none !important; }
.tab-nav { display: none !important; }

/* ── Cards ── */
.stitch-card {
    background: var(--fp-card) !important;
    border: 1px solid var(--fp-border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    margin: 8px 14px !important;
    transition: var(--transition);
}
.stitch-card::before { display: none; }
.stitch-card:hover {
    background: var(--fp-card) !important;
    border-color: var(--fp-border-accent) !important;
    box-shadow: none;
    transform: none;
}

/* ── Form Elements ── */
.stitch-card label, .stitch-card .label-wrap span {
    color: var(--fp-text) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
input[type="number"], input[type="text"], textarea, select,
.stitch-card input, .stitch-card textarea {
    background: var(--fp-surface) !important;
    border: 1px solid var(--fp-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--fp-text) !important;
    padding: 12px 14px !important;
    font-size: 15px !important;
    transition: var(--transition);
}
input:focus, textarea:focus {
    border-color: var(--fp-accent) !important;
    box-shadow: 0 0 0 2px var(--fp-accent-glow) !important;
    outline: none !important;
}

/* ── Buttons ── */
.stitch-btn-primary, .stitch-btn-primary button {
    background: var(--fp-accent) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 15px 28px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    transition: var(--transition);
    text-transform: none !important;
    letter-spacing: 0;
    width: calc(100% - 28px) !important;
    margin: 8px 14px !important;
    box-shadow: none !important;
}
.stitch-btn-primary:hover, .stitch-btn-primary button:hover {
    background: #5ED9D1 !important;
    transform: none !important;
    box-shadow: 0 0 20px var(--fp-accent-glow) !important;
}

.stitch-btn-secondary, .stitch-btn-secondary button {
    background: transparent !important;
    color: var(--fp-text-secondary) !important;
    border: 1px solid var(--fp-border) !important;
    border-radius: var(--radius) !important;
    padding: 13px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: calc(100% - 28px) !important;
    margin: 6px 14px !important;
    transition: var(--transition);
}
.stitch-btn-secondary:hover, .stitch-btn-secondary button:hover {
    background: var(--fp-card) !important;
    border-color: var(--fp-text-muted) !important;
    color: var(--fp-text) !important;
}

/* ── Radio Buttons ── */
.stitch-card .wrap label.selected {
    background: var(--fp-accent) !important;
    color: #000000 !important;
    border-color: var(--fp-accent) !important;
    font-weight: 600 !important;
}

/* ── Badges ── */
.badge-cardio {
    display: inline-block;
    background: var(--fp-green-dim);
    color: var(--fp-green);
    padding: 5px 14px;
    border-radius: var(--radius-pill);
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.3px;
    border: 1px solid rgba(46, 204, 113, 0.2);
}
.badge-strength {
    display: inline-block;
    background: var(--fp-red-dim);
    color: var(--fp-red);
    padding: 5px 14px;
    border-radius: var(--radius-pill);
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.3px;
    border: 1px solid rgba(231, 76, 60, 0.2);
}

/* ── Guide HTML ── */
.guide-container { padding: 2px; }
.guide-container h2 {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--fp-text);
    margin: 14px 0 8px;
}
.guide-container h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--fp-accent);
    margin: 18px 0 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.guide-section {
    background: var(--fp-card-alt);
    border: 1px solid var(--fp-border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    margin: 6px 0;
}
.guide-section ul { list-style: none; padding: 0; }
.guide-section li {
    position: relative;
    padding: 5px 0 5px 18px;
    color: var(--fp-text);
    font-size: 14px;
    line-height: 1.65;
}
.guide-section li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 13px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--fp-accent);
}
.guide-pain {
    background: var(--fp-red-dim);
    border-color: rgba(231, 76, 60, 0.15);
}
.guide-images-wrap {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 10px 0;
}
.guide-images-wrap img {
    max-height: 260px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--fp-border);
    transition: var(--transition);
    object-fit: contain;
}
.guide-images-wrap img:hover {
    border-color: var(--fp-accent);
}

/* ── Heart Rate Zones ── */
.hr-zone-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 3px;
    margin: 8px 0;
}
.hr-zone-table td {
    padding: 10px 12px;
    font-size: 13px;
    background: var(--fp-card-alt);
    border: none;
}
.hr-zone-table td:first-child {
    border-radius: 8px 0 0 8px;
    font-weight: 700;
    width: 80px;
    text-align: center;
}
.hr-zone-table td:nth-child(2) {
    font-weight: 600;
    width: 100px;
    text-align: center;
    color: var(--fp-accent);
}
.hr-zone-table td:last-child {
    border-radius: 0 8px 8px 0;
    color: var(--fp-text-secondary);
    font-size: 12px;
}
.hr-z1 td:first-child { background: #1B5E20; color: #A5D6A7; }
.hr-z2 td:first-child { background: #2E7D32; color: #C8E6C9; }
.hr-z3 td:first-child { background: #E65100; color: #FFE0B2; }
.hr-z4 td:first-child { background: #BF360C; color: #FFCCBC; }
.hr-z5 td:first-child { background: #B71C1C; color: #FFCDD2; }

/* ── YouTube Button ── */
.yt-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--fp-card) !important;
    color: var(--fp-red) !important;
    padding: 11px 20px;
    border-radius: var(--radius-sm);
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    transition: var(--transition);
    margin: 10px 0;
    border: 1px solid rgba(231, 76, 60, 0.25);
}
.yt-btn:hover {
    background: var(--fp-red-dim) !important;
    border-color: var(--fp-red);
}

/* ── Progress Dots ── */
.progress-dots {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin: 14px 0 6px;
}
.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--fp-text-muted);
    opacity: 0.3;
    transition: var(--transition);
}
.dot.active {
    background: var(--fp-accent);
    opacity: 1;
    width: 24px;
    border-radius: 4px;
}

/* ── Tab labels (page indicator text) ── */
.tab-nav button {
    color: var(--fp-text-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: none !important;
    background: none !important;
}
.tab-nav button.selected {
    color: var(--fp-accent) !important;
    font-weight: 600 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--fp-bg); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

/* ── Responsive ── */
@media (max-width: 500px) {
    .gradio-container { max-width: 100% !important; padding: 0 !important; }
    .stitch-card { margin: 6px 10px !important; padding: 16px !important; }
}

/* ── Detection result image ── */
.detection-result img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--fp-border) !important;
}

/* ── Checkbox / Slider ── */
.stitch-card .checkbox-group label {
    color: var(--fp-text-secondary) !important;
    font-size: 13px !important;
}

/* ── Recommendation table (inline HTML) ── */
.guide-section table td {
    background: var(--fp-surface) !important;
    border: 1px solid var(--fp-border) !important;
    color: var(--fp-text) !important;
}
"""


# ═══════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════

def calc_max_hr(age):
    """최대심박수 계산"""
    if age and age > 0:
        return int(220 - age)
    return 0


def build_hr_zones_html(max_hr):
    """유산소 기구용 심박존 테이블 HTML 생성"""
    if max_hr <= 0:
        return ""
    
    zone_ranges = [
        ("Zone 1", 0.50, 0.60, "hr-z1"),
        ("Zone 2", 0.60, 0.70, "hr-z2"),
        ("Zone 3", 0.70, 0.80, "hr-z3"),
        ("Zone 4", 0.80, 0.90, "hr-z4"),
        ("Zone 5", 0.90, 1.00, "hr-z5"),
    ]
    
    html = "<h3>💓 심박 존 가이드</h3>"
    html += "<table class='hr-zone-table'>"
    for zone_name, low, high, css_class in zone_ranges:
        bpm_low = int(max_hr * low)
        bpm_high = int(max_hr * high)
        desc = HR_ZONES.get(zone_name, "")
        html += f"""<tr class='{css_class}'>
            <td>{zone_name}</td>
            <td>{bpm_low}~{bpm_high} BPM</td>
            <td>{desc}</td>
        </tr>"""
    html += "</table>"
    return html

def calc_bmi(weight, height):
    """BMI 계산 (height: cm)"""
    if not weight or not height or height <= 0:
        return 0
    h_m = height / 100
    return round(weight / (h_m * h_m), 1)
 
 
def get_fitness_level(age, gender, bf_pct, bmi):
    """프로필 기반 운동 수준 추정 (초급/중급/상급)"""
    # 체지방율 기반 (입력된 경우 우선)
    if bf_pct and bf_pct > 0:
        if gender == "남성":
            if bf_pct > 25: return "초급"
            elif bf_pct > 18: return "중급"
            else: return "상급"
        else:
            if bf_pct > 32: return "초급"
            elif bf_pct > 25: return "중급"
            else: return "상급"
    # BMI 폴백
    if bmi > 0:
        if bmi > 28: return "초급"
        elif bmi > 23: return "중급"
        else: return "상급"
    return "중급"  # 기본값
 
 
def build_strength_recommendation(gender, weight, fitness_level, ko_name):
    """근력운동 강도/볼륨 권장 HTML"""
    # 피트니스 레벨별 권장 세트/반복/휴식
    configs = {
        "초급": {"sets": "3", "reps": "12~15", "rest": "60~90초", "intensity": "최대 무게의 50~60%", "tip": "정확한 자세 습득이 최우선입니다. 가벼운 무게로 천천히 시작하세요."},
        "중급": {"sets": "4", "reps": "8~12",  "rest": "60~90초", "intensity": "최대 무게의 65~75%", "tip": "점진적 과부하 원칙으로 매주 소폭 무게를 올려보세요."},
        "상급": {"sets": "4~5", "reps": "6~10",  "rest": "90~120초", "intensity": "최대 무게의 75~85%", "tip": "주기화 훈련으로 디로드 주를 포함하세요."},
    }
    cfg = configs.get(fitness_level, configs["중급"])
 
    html = f"<h3>📊 맞춤 운동 강도 ({fitness_level})</h3>"
    html += "<div class='guide-section'>"
    html += f"""
        <table style='width:100%;border-collapse:separate;border-spacing:0 4px;'>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;width:100px;color:var(--fp-accent);'>세트 수</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["sets"]}세트</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>반복 횟수</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["reps"]}회</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>운동 강도</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["intensity"]}</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>세트 간 휴식</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["rest"]}</td>
            </tr>
        </table>
        <p style='font-size:13px;color:var(--fp-text-secondary);margin-top:8px;'>💡 {cfg["tip"]}</p>
    """
    html += "</div>"
    return html
 
 
def build_cardio_recommendation(age, gender, weight, fitness_level, max_hr):
    """유산소 운동 강도/시간/숨찬 정도 권장 HTML"""
    configs = {
        "초급": {
            "zone": "Zone 2 (지방 연소)",
            "hr_range": (0.60, 0.70),
            "duration": "20~30분",
            "frequency": "주 3~4회",
            "rpe": "3~4 / 10",
            "talk_test": "옆 사람과 편하게 대화 가능한 수준. 노래를 부를 수 있다면 강도를 약간 올리세요.",
            "breathing": "코로 마시고 입으로 내쉬기. 4걸음 마시고 4걸음 내쉬는 리듬이 적당합니다.",
            "tip": "심박수를 아래 범위 안에서 유지하세요. '약간 땀이 나고 숨이 차지만 대화가 되는' 상태가 이상적입니다.",
        },
        "중급": {
            "zone": "Zone 3 (심폐 지구력)",
            "hr_range": (0.70, 0.80),
            "duration": "30~45분",
            "frequency": "주 4~5회",
            "rpe": "5~6 / 10",
            "talk_test": "짧은 문장은 가능하지만, 긴 대화는 어려운 수준. 한두 마디 답하기가 적당합니다.",
            "breathing": "리듬이 빨라지며 코와 입 모두 사용. 3걸음 마시고 3걸음 내쉬기.",
            "tip": "인터벌 훈련(2분 빠르게 + 1분 천천히)을 섞으면 심폐 능력 향상에 효과적입니다.",
        },
        "상급": {
            "zone": "Zone 3~4 (심폐~무산소)",
            "hr_range": (0.75, 0.85),
            "duration": "40~60분",
            "frequency": "주 5~6회",
            "rpe": "6~8 / 10",
            "talk_test": "대화가 거의 불가능. 단어 하나 내뱉기도 힘든 구간을 간헐적으로 포함.",
            "breathing": "빠른 호흡. 구간별 다르게 — 저강도 구간에서 호흡 회복 연습.",
            "tip": "HIIT와 LISS를 번갈아 구성하세요. 주 1~2회 고강도 + 나머지는 Zone 2 회복 운동.",
        },
    }
    cfg = configs.get(fitness_level, configs["중급"])
 
    # 심박수 범위 계산
    if max_hr > 0:
        hr_low = int(max_hr * cfg["hr_range"][0])
        hr_high = int(max_hr * cfg["hr_range"][1])
        hr_text = f"{hr_low}~{hr_high} BPM"
    else:
        hr_text = "측정 필요"
 
    # 칼로리 추정 (MET 기반)
    cal_text = ""
    if weight and weight > 0:
        met = 5.0 if fitness_level == "초급" else 7.0 if fitness_level == "중급" else 9.0
        duration_min = 25 if fitness_level == "초급" else 37 if fitness_level == "중급" else 50
        kcal = round(met * weight * (duration_min / 60), 0)
        cal_text = f"약 {int(kcal)} kcal (약 {duration_min}분 기준)"
 
    html = f"<h3>📊 맞춤 유산소 강도 ({fitness_level})</h3>"
    html += "<div class='guide-section'>"
    html += f"""
        <table style='width:100%;border-collapse:separate;border-spacing:0 4px;'>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;width:110px;color:var(--fp-accent);'>권장 심박존</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["zone"]}</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>목표 심박수</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{hr_text}</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>운동 시간</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["duration"]}</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>주간 빈도</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cfg["frequency"]}</td>
            </tr>
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>운동 강도</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>RPE {cfg["rpe"]}</td>
            </tr>
    """
    if cal_text:
        html += f"""
            <tr>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:8px 0 0 8px;font-weight:600;color:var(--fp-accent);'>예상 소모량</td>
                <td style='padding:8px 12px;background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:0 8px 8px 0;'>{cal_text}</td>
            </tr>
        """
    html += "</table>"
 
    # 숨찬 정도 가이드 (Talk Test)
    html += f"""
        <div style='margin-top:12px;padding:12px;background:rgba(0,206,201,0.08);border:1px solid rgba(0,206,201,0.2);border-radius:10px;'>
            <p style='font-size:14px;font-weight:600;color:var(--fp-accent);margin-bottom:6px;'>
                🫁 적절한 숨찬 정도 (Talk Test)
            </p>
            <p style='font-size:13px;color:var(--fp-text);line-height:1.7;'>
                {cfg["talk_test"]}
            </p>
        </div>
        <div style='margin-top:8px;padding:12px;background:rgba(74,123,247,0.08);border:1px solid rgba(74,123,247,0.2);border-radius:10px;'>
            <p style='font-size:14px;font-weight:600;color:var(--fp-accent);margin-bottom:6px;'>
                💨 호흡 리듬 가이드
            </p>
            <p style='font-size:13px;color:var(--fp-text);line-height:1.7;'>
                {cfg["breathing"]}
            </p>
        </div>
        <p style='font-size:13px;color:var(--fp-text-secondary);margin-top:8px;'>💡 {cfg["tip"]}</p>
    """
    html += "</div>"
    return html

# ═══════════════════════════════════════════
# 핵심 기능 함수
# ═══════════════════════════════════════════

def run_detection(image):
    """이미지에서 헬스 기구를 인식하고 결과를 반환"""
    if image is None:
        return None, gr.update(choices=[], value=None), gr.update(interactive=False)
    
    results = model(image, conf=0.25)
    
    try:
        plot_arr = results[0].plot()
        annotated_img = Image.fromarray(plot_arr[..., ::-1].copy())
    except Exception as e:
        print(f"이미지 렌더링 에러: {e}")
        annotated_img = Image.fromarray(image.copy()) if isinstance(image, np.ndarray) else image
    
    detected = {}
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            if cls_name not in detected or conf > detected[cls_name]:
                detected[cls_name] = conf
    
    if not detected:
        print("❌ 인식된 기구 없음")
        return annotated_img, \
               gr.update(choices=["인식된 기구가 없습니다"], value="인식된 기구가 없습니다"), \
               gr.update(interactive=False)
    
    choices = []
    for eng_name, conf in sorted(detected.items(), key=lambda x: -x[1]):
        ko = EN_TO_KO.get(eng_name, eng_name)
        label = f"{ko} ({eng_name}) — {conf:.0%}"
        choices.append(label)
    
    print(f"✅ 처리가 완료되었습니다. (인식된 기구: {choices})")
    
    return annotated_img, \
           gr.update(choices=choices, value=choices[0]), \
           gr.update(interactive=True)

def build_guide_html(selected_equip_label, selected_mode_name, age, gender, weight, height, bf_pct):
    """운동 가이드 HTML 생성 (프로필 기반 맞춤 권장 포함)"""
    if not selected_equip_label or "인식된 기구가 없습니다" in selected_equip_label:
        return "<p style='color:var(--fp-text-secondary);text-align:center;'>기구를 먼저 선택해 주세요.</p>"
 
    eng_name = selected_equip_label.split("(")[1].split(")")[0].strip()
    equip = EQUIPMENT.get(eng_name)
    if not equip:
        return f"<p>'{eng_name}' 기구 정보를 찾을 수 없습니다.</p>"
 
    category = equip.get("category", "")
    ko_name = equip.get("ko_name", eng_name)
    is_cardio = "유산소" in category
 
    html = "<div class='guide-container'>"
    badge_cls = "badge-cardio" if is_cardio else "badge-strength"
    html += f"<span class='{badge_cls}'>{category}</span>"
    html += f"<h2>🏋️ {ko_name}</h2>"
 
    # ── 프로필 계산 ──
    max_hr = calc_max_hr(age)
    bmi = calc_bmi(weight, height)
    fitness_level = get_fitness_level(age, gender, bf_pct, bmi)
 
    # ── 프로필 요약 (BMI + 운동수준 추가) ──
    profile_parts = []
    if age: profile_parts.append(f"나이: {int(age)}세")
    if gender: profile_parts.append(f"성별: {gender}")
    if weight: profile_parts.append(f"체중: {weight}kg")
    if height: profile_parts.append(f"신장: {height}cm")
    if bf_pct and bf_pct > 0: profile_parts.append(f"체지방: {bf_pct}%")
    if bmi > 0: profile_parts.append(f"BMI: {bmi}")
    if max_hr: profile_parts.append(f"최대심박수: {max_hr} BPM")
    profile_parts.append(f"추정 운동수준: {fitness_level}")
 
    if profile_parts:
        html += "<div class='guide-section' style='margin-top:12px;'>"
        html += "<h3 style='margin-top:0;'>👤 내 프로필</h3>"
        html += "<p style='color:var(--fp-text-secondary);font-size:13px;'>" + " · ".join(profile_parts) + "</p>"
        html += "</div>"
 
    # ── 유산소/근력 분기 ──
    if is_cardio:
        # 맞춤 유산소 강도 (심박존 테이블 대체)
        html += build_cardio_recommendation(age, gender, weight, fitness_level, max_hr)
 
        stretching = equip.get("pre_post_stretching", {})
        if stretching:
            html += "<h3>🧘 운동 전후 스트레칭</h3>"
            html += "<div class='guide-section'><ul>"
            for part, instruction in stretching.items():
                html += f"<li><strong>{part}</strong>: {instruction}</li>"
            html += "</ul></div>"
    else:
        setup = equip.get("machine_setup", [])
        if setup:
            html += "<h3>⚙️ 머신 세팅</h3>"
            html += "<div class='guide-section'><ul>"
            for item in setup:
                html += f"<li>{item}</li>"
            html += "</ul></div>"
 
        # 맞춤 근력 강도
        html += build_strength_recommendation(gender, weight, fitness_level, ko_name)
 
    # ── 통증 대처 ──
    pain = equip.get("pain_management", {})
    if pain:
        html += "<h3>🩹 통증 대처법</h3>"
        html += "<div class='guide-section guide-pain'><ul>"
        for symptom, remedy in pain.items():
            html += f"<li><strong>{symptom.replace('_', ' ').title()}</strong>: {remedy}</li>"
        html += "</ul></div>"
 
    # ── 운동 모드 상세 ──
    modes = equip.get("exercise_modes", [])
    if selected_mode_name and modes:
        mode = next((m for m in modes if m["mode_name"] == selected_mode_name), modes[0])
        html += f"<h3>🎯 운동 모드: {mode['mode_name']}</h3>"
 
        breathing = mode.get("breathing", "")
        if breathing:
            html += "<div class='guide-section'>"
            html += f"<p style='font-size:14px;'>💨 <strong>호흡법</strong>: {breathing}</p>"
            html += "</div>"
 
        guide_points = mode.get("guide", [])
        if guide_points:
            html += "<div class='guide-section'><ul>"
            for pt in guide_points:
                html += f"<li>{pt}</li>"
            html += "</ul></div>"
 
        # ── 이미지 (base64 인코딩) ──
        imgs = mode.get("exercise_images", [])
        if imgs:
            html += "<h3>📸 정자세 가이드</h3>"
            html += "<div class='guide-images-wrap'>"
            for img_file in imgs:
                img_path = IMAGE_DIR / img_file
                if img_path.exists():
                    mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    html += f"""<div style='text-align:center;'>
                        <img src='data:{mime};base64,{b64}'
                             style='max-height:280px;border-radius:10px;border:2px solid rgba(74,123,247,0.2);'>
                    </div>"""
                else:
                    html += "<p style='color:var(--fp-text-secondary);font-size:12px;'>이미지 준비 중</p>"
            html += "</div>"
 
        yt_url = mode.get("youtube_url", "")
        if yt_url:
            html += f"""<a href='{yt_url}' target='_blank' class='yt-btn'>
                ▶ YouTube에서 영상 보기
            </a>"""
 
    html += "</div>"
    return html


def get_mode_choices(selected_equip_label):
    """선택된 기구의 운동 모드 리스트 반환"""
    if not selected_equip_label or "인식된 기구가 없습니다" in selected_equip_label:
        return gr.update(choices=[], value=None, visible=False)
    
    eng_name = selected_equip_label.split("(")[1].split(")")[0].strip()
    equip = EQUIPMENT.get(eng_name)
    if not equip:
        return gr.update(choices=[], value=None, visible=False)
    
    modes = equip.get("exercise_modes", [])
    mode_names = [m["mode_name"] for m in modes]
    
    if not mode_names:
        return gr.update(choices=[], value=None, visible=False)
    
    return gr.update(choices=mode_names, value=mode_names[0], visible=True)


# ═══════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════

def create_app():
    with gr.Blocks(title="GymBuddy — 헬스 기구 인식 가이드") as demo:
        
        # ── 상단 헤더 ──
        gr.HTML("""
        <div class='stitch-header'>
            <span class='stitch-emoji'>🏋️‍♂️</span>
            <h1>GymBuddy</h1>
            <p>AI 헬스 기구 인식 & 운동 가이드</p>
        </div>
        """)
        
        # ── 진행 상태 표시 (HTML로 동적 변경) ──
        progress_html = gr.HTML(
            "<div class='progress-dots'>"
            "<div class='dot active'></div>"
            "<div class='dot'></div>"
            "<div class='dot'></div>"
            "</div>"
        )
        
        # ── 탭 구조 ──
        with gr.Tabs() as tabs:
            
            # ═══ Page 0: 사용자 프로필 ═══
            with gr.Tab("프로필", id=0) as tab0:
                gr.HTML("""
                <div style='text-align:center;margin:8px 12px 4px;'>
                    <h2 style='font-family:Inter;font-size:20px;color:var(--fp-text);'>
                        내 프로필 설정
                    </h2>
                    <p style='color:var(--fp-text-secondary);font-size:13px;margin-top:2px;'>
                        맞춤형 운동 가이드를 위해 기본 정보를 입력해 주세요
                    </p>
                </div>
                """)
                
                with gr.Group(elem_classes="stitch-card"):
                    age_input = gr.Number(
                        label="🎂 나이",
                        value=30,
                        minimum=10, maximum=100,
                        precision=0,
                        info="최대심박수 계산에 사용됩니다 (220-나이)"
                    )
                    gender_input = gr.Radio(
                        label="👤 성별",
                        choices=["남성", "여성"],
                        value="남성"
                    )
                    weight_input = gr.Number(
                        label="⚖️ 체중 (kg)",
                        value=70,
                        minimum=30, maximum=200,
                        precision=1
                    )
                    height_input = gr.Number(
                        label="📏 신장 (cm)",
                        value=175,
                        minimum=100, maximum=250,
                        precision=1
                    )
                
                with gr.Group(elem_classes="stitch-card"):
                    gr.HTML("""
                    <p style='color:var(--fp-text-secondary);font-size:12px;margin-bottom:8px;'>
                        💡 체지방률은 선택 입력입니다. 모르시면 0으로 두세요.
                    </p>
                    """)
                    bf_input = gr.Slider(
                        label="📊 체지방률 (%)",
                        minimum=0, maximum=50,
                        value=0, step=0.5,
                        info="선택 사항 — 0이면 표시하지 않습니다"
                    )
                
                next_btn_0 = gr.Button(
                    "다음 → 기구 인식",
                    elem_classes="stitch-btn-primary",
                    size="lg"
                )
            
            # ═══ Page 1: 기구 인식 ═══
            with gr.Tab("기구 인식", id=1) as tab1:
                gr.HTML("""
                <div style='text-align:center;margin:8px 12px 4px;'>
                    <h2 style='font-family:Inter;font-size:20px;color:var(--fp-text);'>
                        헬스 기구 인식
                    </h2>
                    <p style='color:var(--fp-text-secondary);font-size:13px;margin-top:2px;'>
                        사진을 업로드하거나 웹캠으로 촬영하세요
                    </p>
                </div>
                """)
                
                with gr.Group(elem_classes="stitch-card"):
                    input_image = gr.Image(
                        sources=["upload", "webcam"],
                        type="numpy",
                        label="📷 헬스 기구 사진",
                        height=300
                    )
                
                detect_btn = gr.Button(
                    "🔍 기구 인식하기",
                    elem_classes="stitch-btn-primary",
                    size="lg"
                )
                
                with gr.Group(elem_classes="stitch-card detection-result", visible=True):
                    result_image = gr.Image(
                        label="인식 결과",
                        type="pil",
                        interactive=False,
                        height=300
                    )
                
                with gr.Group(elem_classes="stitch-card"):
                    equip_radio = gr.Radio(
                        label="🏋️ 인식된 기구 목록",
                        choices=[],
                        visible=True,
                        interactive=True
                    )
                
                guide_btn = gr.Button(
                    "운동 가이드 보기 →",
                    elem_classes="stitch-btn-primary",
                    visible=True,
                    interactive=False,
                    size="lg"
                )
                
                back_btn_1 = gr.Button(
                    "← 프로필로 돌아가기",
                    elem_classes="stitch-btn-secondary"
                )
            
            # ═══ Page 2: 운동 가이드 ═══
            with gr.Tab("운동 가이드", id=2) as tab2:
                gr.HTML("""
                <div style='text-align:center;margin:8px 12px 4px;'>
                    <h2 style='font-family:Inter;font-size:20px;color:var(--fp-text);'>
                        운동 가이드
                    </h2>
                    <p style='color:var(--fp-text-secondary);font-size:13px;margin-top:2px;'>
                        선택한 기구의 상세 운동 가이드입니다
                    </p>
                </div>
                """)
                
                with gr.Group(elem_classes="stitch-card"):
                    mode_radio = gr.Radio(
                        label="🎯 운동 모드 선택",
                        choices=[],
                        visible=False,
                        interactive=True
                    )
                
                with gr.Group(elem_classes="stitch-card"):
                    guide_html = gr.HTML("")
                
                back_btn_2 = gr.Button(
                    "← 기구 인식으로 돌아가기",
                    elem_classes="stitch-btn-secondary"
                )
        
        # ═══════════════════════════════════
        # 이벤트 핸들러
        # ═══════════════════════════════════
        
        # Page 0 → Page 1
        def go_to_page1():
            return (
                gr.update(selected=1),
                "<div class='progress-dots'>"
                "<div class='dot'></div>"
                "<div class='dot active'></div>"
                "<div class='dot'></div>"
                "</div>"
            )
        
        next_btn_0.click(
            fn=go_to_page1,
            outputs=[tabs, progress_html]
        )
        
        # Page 1 → Page 0
        def go_to_page0():
            return (
                gr.update(selected=0),
                "<div class='progress-dots'>"
                "<div class='dot active'></div>"
                "<div class='dot'></div>"
                "<div class='dot'></div>"
                "</div>"
            )
        
        back_btn_1.click(
            fn=go_to_page0,
            outputs=[tabs, progress_html]
        )
        
        # 기구 인식 실행
        detect_btn.click(
            fn=run_detection,
            inputs=[input_image],
            outputs=[result_image, equip_radio, guide_btn]
        )
        
        # Page 1 → Page 2 (운동 가이드 보기)
        def go_to_page2(equip_label, age, gender, weight, height, bf_pct):
            modes = get_mode_choices(equip_label)
            
            # 첫 번째 모드로 초기 가이드 생성
            first_mode = None
            if equip_label and "인식된 기구가 없습니다" not in equip_label:
                eng_name = equip_label.split("(")[1].split(")")[0].strip()
                equip = EQUIPMENT.get(eng_name)
                if equip:
                    ex_modes = equip.get("exercise_modes", [])
                    if ex_modes:
                        first_mode = ex_modes[0]["mode_name"]
            
            guide = build_guide_html(equip_label, first_mode, age, gender, weight, height, bf_pct)
            
            progress = (
                "<div class='progress-dots'>"
                "<div class='dot'></div>"
                "<div class='dot'></div>"
                "<div class='dot active'></div>"
                "</div>"
            )
            
            return gr.update(selected=2), progress, modes, guide
        
        guide_btn.click(
            fn=go_to_page2,
            inputs=[equip_radio, age_input, gender_input, weight_input, height_input, bf_input],
            outputs=[tabs, progress_html, mode_radio, guide_html]
        )
        
        # Page 2 → Page 1
        def go_back_to_page1():
            return (
                gr.update(selected=1),
                "<div class='progress-dots'>"
                "<div class='dot'></div>"
                "<div class='dot active'></div>"
                "<div class='dot'></div>"
                "</div>"
            )
        
        back_btn_2.click(
            fn=go_back_to_page1,
            outputs=[tabs, progress_html]
        )
        
        # 모드 변경 시 가이드 업데이트
        mode_radio.change(
            fn=build_guide_html,
            inputs=[equip_radio, mode_radio, age_input, gender_input,
                    weight_input, height_input, bf_input],
            outputs=[guide_html]
        )
    
    return demo


# ═══════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════

if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        allowed_paths=[str(IMAGE_DIR)],
        css=STITCH_CSS,
        share=False
    )