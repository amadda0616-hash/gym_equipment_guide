# Phase 5: 모바일 앱 구축 매뉴얼 (Windows / WSL 환경)

## 환경 제약 사항

| | macOS | Windows/WSL |
|---|---|---|
| Xcode (Swift 네이티브) | ✅ | ❌ 불가 |
| Flutter (크로스 플랫폼) | ✅ | ✅ Android만 빌드, iOS는 Mac 필요 |
| React Native (Expo) | ✅ | ✅ Android만 빌드, iOS는 Mac 필요 |
| **PWA (웹앱)** | ✅ | **✅ 완전 지원** |

> **결론**: Windows/WSL에서 iPhone에서 동작하는 앱을 만드는 가장 현실적인 방법은 **PWA(Progressive Web App)** 또는 **FastAPI + 모바일 웹**입니다.
> 현재 Gradio app_fitpro.py가 이미 모바일 퍼스트 디자인이므로, 이를 발전시키는 방향이 가장 효율적입니다.

---

## 선택지 비교

### Option 1: Gradio 앱을 외부 공개 (가장 빠름, 5분)

현재 app_fitpro.py를 그대로 iPhone에서 사용합니다.

```bash
# share=True로 실행하면 Gradio가 공개 URL 생성
python app_fitpro.py  # launch()에서 share=True 설정
```

출력 예시:
```
* Running on local URL:  http://127.0.0.1:7869
* Running on public URL: https://abc123def456.gradio.live  ← iPhone에서 이 URL 접속
```

**장점**: 코드 수정 0줄, 즉시 사용 가능
**단점**: 72시간 후 URL 만료, Gradio UI 제약, 앱스토어 배포 불가

### Option 2: FastAPI + PWA 웹앱 (권장, 1~2일)

서버(server.py) + 모바일 최적화 웹 프론트엔드를 만들어 iPhone Safari에서 "홈 화면에 추가"하면 네이티브 앱처럼 동작합니다.

**장점**: Windows/WSL에서 완전 개발 가능, 카메라 접근 가능, 앱처럼 동작
**단점**: 앱스토어 배포 불가 (PWA는 웹 기반)

### Option 3: Flutter 크로스 플랫폼 (iOS 빌드 시 Mac 필요)

Windows에서 개발은 가능하지만, iOS 빌드/테스트는 Mac이 필요합니다.
Codemagic 같은 클라우드 빌드 서비스로 우회 가능하지만 설정이 복잡합니다.

---

## Option 2 상세: FastAPI + PWA 구축

### 아키텍처

```
┌─────────────────────────────────────────┐
│  iPhone Safari (PWA)                     │
│  ┌────────────────────────────────────┐  │
│  │  HTML + CSS + JavaScript            │  │
│  │  - 카메라 촬영 (MediaDevices API)   │  │
│  │  - 프로필 입력                      │  │
│  │  - 운동 가이드 표시                 │  │
│  └──────────┬─────────────────────────┘  │
└─────────────┼────────────────────────────┘
              │ fetch() API 호출
              ▼
┌─────────────────────────────────────────┐
│  Windows PC / WSL                        │
│  ┌────────────────────────────────────┐  │
│  │  FastAPI (server.py)               │  │
│  │  + YOLO26 모델 (best.pt)           │  │
│  │  + 메타데이터 (equipment_guide.json)│  │
│  │  + 정적 파일 서빙 (index.html)     │  │
│  └────────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 디렉토리 구조

```
C:\Users\user\github\yolo26_app\gym\data\
├── dataset/
│   ├── app_fitpro.py              # Gradio 앱
│   ├── server.py                  # FastAPI + PWA 서버
│   ├── equipment_guide_v4.json    # 메타데이터
│   ├── data.yaml
│   ├── static/                    # ← 새로 생성 (PWA 프론트엔드)
│   │   ├── index.html
│   │   ├── manifest.json
│   │   └── sw.js
│   ├── train/ valid/ test/
│   └── runs/detect/...            # 학습 결과
└── images/                        # ← dataset/ 밖, data/ 아래
    ├── cardio/
    └── strength/
```

### Step 1: server.py 설정 (정적 파일 서빙 내장)

server.py에 PWA 정적 파일 서빙이 이미 통합되어 있습니다.
`static/` 폴더가 server.py와 같은 디렉토리에 있으면 자동으로 마운트됩니다.

API 엔드포인트는 모두 `/api/` 접두사를 사용합니다:
- `POST /api/detect` — 기구 인식
- `POST /api/detect/annotated` — bbox 이미지 반환
- `GET /api/health` — 서버 상태
- `GET /` — PWA 프론트엔드 (static/index.html)

```bash
# 서버 실행 (static/ 폴더가 있으면 PWA 자동 서빙)
python server.py

# 다른 모델로 실행
python server.py --model runs/detect/gym_yolo26m_100ep/weights/best.pt
```

### Step 2: static/index.html 생성

이 파일이 iPhone에서 열리는 앱 화면입니다. 단일 HTML 파일에 CSS + JS를 모두 포함합니다.



### Step 3: PWA 매니페스트

**static/manifest.json**:
```json
{
  "name": "GymBuddy",
  "short_name": "GymBuddy",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#000000",
  "theme_color": "#000000",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" }
  ]
}
```

**static/sw.js** (서비스 워커 — 최소 구현):
```javascript
self.addEventListener('fetch', event => {
  event.respondWith(fetch(event.request));
});
```

### Step 4: 앱 아이콘

192x192px PNG 파일을 `static/icon-192.png`으로 저장합니다.
간단하게 만들려면 이모지를 캡처하거나, Canva에서 무료로 생성.

---

## 실행 및 테스트

### 로컬 테스트 (PC 브라우저)

```bash
# 서버 실행
python server.py

# 브라우저에서 접속
# http://localhost:8000
```

### iPhone 테스트 (같은 Wi-Fi)

```bash
# 1. PC의 로컬 IP 확인
ipconfig  # Windows → IPv4 주소 확인 (예: 192.168.0.10)

# 2. 외부 접근 허용하여 서버 실행
python server.py --host 0.0.0.0

# 3. iPhone Safari에서 접속
# http://192.168.0.10:8000
```

### iPhone 홈 화면에 추가 (앱처럼 사용)

1. Safari에서 `http://192.168.0.10:8000` 접속
2. 하단 공유 버튼 (↑ 아이콘) 탭
3. "홈 화면에 추가" 선택
4. "GymBuddy" 이름 확인 후 추가
5. 홈 화면에 앱 아이콘이 생성됨 → 탭하면 전체 화면으로 실행

---

## 외부 공개 (ngrok 사용)

같은 Wi-Fi가 아닌 환경에서 테스트하려면:

```bash
# 1. ngrok 설치
# https://ngrok.com/download (Windows용 다운로드)

# 2. 서버 실행
python server.py --host 0.0.0.0

# 3. 다른 터미널에서 ngrok 실행
ngrok http 8000

# 출력 예시:
# Forwarding https://abc123.ngrok-free.app → http://localhost:8000
# → iPhone에서 이 HTTPS URL로 접속
```

> ngrok의 HTTPS URL은 카메라 권한이 자동으로 허용됩니다 (HTTP는 Safari에서 카메라 차단).

---

## 모델 교체 방법

```bash
# 서버만 재시작하면 됨. 프론트엔드 코드 수정 불필요.
python server.py --model runs/detect/gym_yolo26m_100ep/weights/best.pt
```

---

## 체크리스트

- [ ] `pip install fastapi uvicorn python-multipart pillow ultralytics`
- [ ] `static/` 폴더 생성 + `index.html`, `manifest.json`, `sw.js` 배치
- [ ] `python server.py` 실행
- [ ] PC 브라우저에서 `http://localhost:8000` 접속 확인
- [ ] iPhone에서 `http://PC_IP:8000` 접속 확인
- [ ] 카메라 촬영 → 기구 인식 → 운동 가이드 표시 확인
- [ ] 홈 화면에 추가하여 앱처럼 동작 확인
