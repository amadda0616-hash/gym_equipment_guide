"""
Phase 5: 헬스 기구 인식 — FastAPI 백엔드 + PWA 서버
YOLO26 추론 API + 정적 파일(PWA 프론트엔드) 서빙을 통합합니다.

사용법:
    python server.py                          # 기본 실행 (port 8000)
    python server.py --port 8080              # 포트 변경
    python server.py --model path/to/best.pt  # 모델 변경

접속:
    PC 브라우저:  http://localhost:8000
    iPhone:      http://PC_IP:8000 (같은 Wi-Fi)
    Swagger UI:  http://localhost:8000/docs

API 엔드포인트:
    POST /api/detect          — 이미지 업로드 → 기구 인식 + 가이드
    POST /api/detect/annotated — bbox 그려진 이미지 반환
    GET  /api/equipment       — 전체 기구 메타데이터
    GET  /api/equipment/{name} — 특정 기구 메타데이터
    GET  /api/health          — 서버 상태 확인
    GET  /                    — PWA 프론트엔드 (static/index.html)
"""

import argparse
import json
import time
import base64
import mimetypes
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# ─────────────────────────────────────────
# 설정 (CLI 인자 또는 기본값)
# ─────────────────────────────────────────
DEFAULT_MODEL = Path(__file__).parent / "runs" / "detect" / "gym_yolo26s_100ep" / "weights" / "best.pt"
DEFAULT_JSON  = Path(__file__).parent / "equipment_guide_v4.json"
DEFAULT_IMAGE_DIR = (Path(__file__).parent.parent / "images").resolve()
STATIC_DIR = Path(__file__).parent / "static"

# ─────────────────────────────────────────
# 전역 변수 (startup에서 초기화)
# ─────────────────────────────────────────
model = None
equipment_db = None
heart_rate_zones = None
class_names = None
image_dir = None


def load_model(model_path: str):
    """YOLO 모델 로드"""
    from ultralytics import YOLO
    global model, class_names
    model = YOLO(model_path)
    class_names = model.names
    print(f"✅ 모델 로드: {model_path}")
    print(f"   클래스 수: {len(class_names)}")


def load_metadata(json_path: str):
    """메타데이터 JSON 로드"""
    global equipment_db, heart_rate_zones
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    equipment_db = data["equipment_list"]
    heart_rate_zones = data.get("heart_rate_zones", {})
    print(f"✅ 메타데이터 로드: {json_path}")
    print(f"   기구 수: {len(equipment_db)}")


# ─────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────
def calc_max_hr(age):
    if age and age > 0:
        return int(220 - age)
    return 0


def calc_bmi(weight, height):
    if not weight or not height or height <= 0:
        return 0
    return round(weight / (height / 100) ** 2, 1)


def get_fitness_level(age, gender, bf_pct, bmi):
    if bf_pct and bf_pct > 0:
        if gender == "남성":
            if bf_pct > 25: return "초급"
            elif bf_pct > 18: return "중급"
            else: return "상급"
        else:
            if bf_pct > 32: return "초급"
            elif bf_pct > 25: return "중급"
            else: return "상급"
    if bmi > 0:
        if bmi > 28: return "초급"
        elif bmi > 23: return "중급"
        else: return "상급"
    return "중급"


def get_strength_recommendation(fitness_level):
    configs = {
        "초급": {"sets": "3세트", "reps": "12~15회", "rest": "60~90초", "intensity": "최대 무게의 50~60%", "tip": "정확한 자세 습득이 최우선입니다."},
        "중급": {"sets": "4세트", "reps": "8~12회", "rest": "60~90초", "intensity": "최대 무게의 65~75%", "tip": "점진적 과부하 원칙으로 매주 소폭 무게를 올려보세요."},
        "상급": {"sets": "4~5세트", "reps": "6~10회", "rest": "90~120초", "intensity": "최대 무게의 75~85%", "tip": "주기화 훈련으로 디로드 주를 포함하세요."},
    }
    return configs.get(fitness_level, configs["중급"])


def get_cardio_recommendation(fitness_level, max_hr):
    configs = {
        "초급": {
            "zone": "Zone 2 (지방 연소)", "hr_range": [0.60, 0.70],
            "duration": "20~30분", "frequency": "주 3~4회", "rpe": "3~4/10",
            "talk_test": "옆 사람과 편하게 대화 가능한 수준",
            "breathing_guide": "코로 마시고 입으로 내쉬기. 4걸음 마시고 4걸음 내쉬기.",
        },
        "중급": {
            "zone": "Zone 3 (심폐 지구력)", "hr_range": [0.70, 0.80],
            "duration": "30~45분", "frequency": "주 4~5회", "rpe": "5~6/10",
            "talk_test": "짧은 문장은 가능하지만 긴 대화는 어려운 수준",
            "breathing_guide": "코와 입 모두 사용. 3걸음 마시고 3걸음 내쉬기.",
        },
        "상급": {
            "zone": "Zone 3~4 (심폐~무산소)", "hr_range": [0.75, 0.85],
            "duration": "40~60분", "frequency": "주 5~6회", "rpe": "6~8/10",
            "talk_test": "대화가 거의 불가능한 구간을 간헐적으로 포함",
            "breathing_guide": "빠른 호흡. 저강도 구간에서 호흡 회복 연습.",
        },
    }
    cfg = configs.get(fitness_level, configs["중급"])
    if max_hr > 0:
        cfg["target_hr"] = {
            "low": int(max_hr * cfg["hr_range"][0]),
            "high": int(max_hr * cfg["hr_range"][1]),
        }
    return cfg


def get_exercise_image_base64(img_file: str) -> str:
    """운동 이미지를 base64로 인코딩"""
    if not image_dir:
        return None
    img_path = image_dir / img_file
    if img_path.exists():
        mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    return None


# ─────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────
app = FastAPI(
    title="GymBuddy API",
    description="헬스 기구 인식 & 운동 가이드 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(class_names) if class_names else 0,
        "equipment_count": len(equipment_db) if equipment_db else 0,
    }


@app.get("/api/equipment")
async def get_all_equipment():
    """전체 기구 메타데이터 반환"""
    return {"equipment_list": equipment_db, "heart_rate_zones": heart_rate_zones}


@app.get("/api/equipment/{name}")
async def get_equipment(name: str):
    """특정 기구 메타데이터 반환"""
    equip = equipment_db.get(name)
    if not equip:
        raise HTTPException(status_code=404, detail=f"Equipment '{name}' not found")
    return {name: equip}


@app.post("/api/detect")
async def detect_equipment(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.1, le=0.9),
    age: int = Query(None),
    gender: str = Query(None),
    weight: float = Query(None),
    height: float = Query(None),
    body_fat: float = Query(None),
    mode_name: str = Query(None),
):
    """
    이미지 업로드 → YOLO 추론 → 기구 인식 + 맞춤 운동 가이드 반환
    
    - file: 이미지 파일
    - conf: 신뢰도 임계값 (기본 0.25)
    - age, gender, weight, height, body_fat: 사용자 프로필 (선택)
    - mode_name: 특정 운동 모드 선택 (선택, 없으면 첫 번째 모드)
    """
    start_time = time.time()
    
    # 이미지 읽기
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 읽기 실패: {str(e)}")
    
    # YOLO 추론
    results = model(img_array, conf=conf)
    
    # 탐지 결과 파싱
    detections = []
    detected_map = {}  # 중복 제거용
    
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            if cls_name not in detected_map or confidence > detected_map[cls_name]["confidence"]:
                detected_map[cls_name] = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(v, 1) for v in bbox],
                }
    
    detections = sorted(detected_map.values(), key=lambda x: -x["confidence"])
    
    # 프로필 기반 계산
    max_hr = calc_max_hr(age)
    bmi = calc_bmi(weight, height)
    fitness_level = get_fitness_level(age, gender, body_fat, bmi)
    
    profile = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "body_fat": body_fat,
        "bmi": bmi,
        "max_hr": max_hr,
        "fitness_level": fitness_level,
    }
    
    # 각 탐지된 기구에 가이드 정보 매핑
    guides = []
    for det in detections:
        cls_name = det["class_name"]
        equip = equipment_db.get(cls_name, {})
        
        if not equip:
            continue
        
        ko_name = equip.get("ko_name", cls_name)
        category = equip.get("category", "")
        is_cardio = "유산소" in category
        
        # 운동 모드 처리
        modes = equip.get("exercise_modes", [])
        selected_mode = None
        if mode_name:
            selected_mode = next((m for m in modes if m["mode_name"] == mode_name), None)
        if not selected_mode and modes:
            selected_mode = modes[0]
        
        # 운동 모드 이미지 base64 변환
        mode_list = []
        for m in modes:
            mode_data = {
                "mode_name": m["mode_name"],
                "breathing": m.get("breathing", ""),
                "guide": m.get("guide", []),
                "youtube_url": m.get("youtube_url", ""),
            }
            # 이미지 base64 인코딩
            img_list = []
            for img_file in m.get("exercise_images", []):
                b64 = get_exercise_image_base64(img_file)
                if b64:
                    img_list.append(b64)
            mode_data["exercise_images_base64"] = img_list
            mode_list.append(mode_data)
        
        # 맞춤 추천
        recommendation = None
        if is_cardio:
            recommendation = get_cardio_recommendation(fitness_level, max_hr)
        else:
            recommendation = get_strength_recommendation(fitness_level)
        
        guide = {
            "class_name": cls_name,
            "ko_name": ko_name,
            "category": category,
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "machine_setup": equip.get("machine_setup", []),
            "pre_post_stretching": equip.get("pre_post_stretching", {}),
            "pain_management": equip.get("pain_management", {}),
            "exercise_modes": mode_list,
            "recommendation": recommendation,
        }
        guides.append(guide)
    
    elapsed = round(time.time() - start_time, 3)
    
    return {
        "success": True,
        "processing_time_sec": elapsed,
        "detections_count": len(detections),
        "profile": profile,
        "guides": guides,
    }


# ─────────────────────────────────────────
# Annotated Image 엔드포인트 (선택)
# ─────────────────────────────────────────
@app.post("/api/detect/annotated")
async def detect_with_annotated_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.1, le=0.9),
):
    """이미지 업로드 → bbox가 그려진 이미지를 base64로 반환"""
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 읽기 실패: {str(e)}")
    
    results = model(img_array, conf=conf)
    
    try:
        plot_arr = results[0].plot()
        annotated = Image.fromarray(plot_arr[..., ::-1])
    except Exception:
        annotated = img
    
    # base64 인코딩
    buffer = BytesIO()
    annotated.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "annotated_image": f"data:image/jpeg;base64,{b64}",
    }


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GymBuddy API + PWA Server")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help=f"모델 경로 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON),
                        help=f"메타데이터 JSON 경로 (기본: {DEFAULT_JSON})")
    parser.add_argument("--images", type=str, default=str(DEFAULT_IMAGE_DIR),
                        help=f"운동 이미지 디렉토리 (기본: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    
    args = parser.parse_args()
    
    image_dir = Path(args.images)
    load_model(args.model)
    load_metadata(args.json)
    
    # PWA 정적 파일 마운트 (static/ 폴더가 있을 때만)
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
        print(f"✅ PWA 프론트엔드: {STATIC_DIR.resolve()}")
    else:
        print(f"⚠️ static/ 폴더 없음 — API 전용 모드 (PWA 없이 실행)")
        print(f"   PWA를 사용하려면 {STATIC_DIR} 폴더에 index.html을 배치하세요")
    
    print(f"\n🚀 GymBuddy 서버 시작")
    print(f"   앱:        http://{args.host}:{args.port}")
    print(f"   API 문서:  http://{args.host}:{args.port}/docs")
    print(f"   모델:      {args.model}")
    
    uvicorn.run(app, host=args.host, port=args.port)
