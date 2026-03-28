# gym_equipment_guide
> 헬스장 장비 인식 후 사용설명 제공 서비스 프로젝트

# 📃 Contents

[1. 프로젝트 소개](#1-프로젝트-소개) <br>
  - [목표](#목표)
  - [데이터셋 주소](#데이터셋-주소)
  
[2. 파이프라인](#2-파이프라인) <br>
  - [phase 0: 데이터 정제](#phase-0:-데이터-정제)
  - [phase 1: EDA + 전처리](#phase-1:-EDA-+-전처리)
  - [phase 2: 실험 (YOLO26 학습)](#phase-2:-실험)
  - [phase 3: 결과 (성능 검증)](#phase-3:-결과)
  - [phase 4: Gradio 웹서비스](#phase-4:-Gradio-웹서비스)
  - [phase 5: ios 앱 서비스](#phase-5:-io-앱-서비스) 

[3. 데이터](#3-데이터) <br>
  - [EDA 요약](#eda-요약)

[4. 실험](#3-실험) <br>
  - [baseline](#0-baseline)

[5. 결과](#4-결과) <br>

[6. 프로젝트 회고](#5-프로젝트-회고) <br>
  - [어려웠던 점](#어려웠던-점)
  - [보완할 점](#보완할-점)


# 1. 프로젝트 소개

### 목표

- 사용자 프로필 입력(연령, 성별, 체중(kg), 신장(cm), 체지방율(%, 스킵 가능) 입력)
- Roboflow의 운동 장비 데이터셋을 활용하여 YOLO26으로 기구인식(image detection)
- 인식된 기구 목록을 라디오 버튼으로 표시 (한글명 + 영문명 + 신뢰도%)
- 기구 선택 후 "운동 가이드 보기" 클릭 시 Page 2로 이동
- Page 2 — 운동 가이드 상세 (입력된 사용자 프로필에 따라 운동량과 운동 강도 추천)
- 상단부터 순서대로:
1. **category** (유산소/근력) 배지
2. **ko_name** (한글 기구명)
3. **machine_setup** (근력 기구) 또는 **pre_post_stretching** (유산소 기구)
4. **pain_management** (통증 대처법)
5. **exercise_modes 선택 UI** — 라디오 버튼으로 모드 선택
6. 선택된 모드의 상세 정보:
   - **mode_name** (모드명)
   - **breathing** (호흡법)
   - **guide** (자세 포인트)
   - **exercise_images** (정자세 가이드 이미지)
   - **youtube_url** (운동 영상 링크 — 현재 공란이면 버튼 미표시)
  
7. 위 가이드를 gradio를 이용해 웹서비스를 제작하고 app을 통해 온디바이스로도 사용가능하게 제작

### 데이터셋 주소

1. Fitness Equipment Recognition Computer Vision Model 
https://universe.roboflow.com/fitness-equipment-recognition-colin-pruzek/fitness-equipment-recognition-wlluo/dataset/3
2. fitness_equipments Computer Vision Dataset
https://universe.roboflow.com/athenazhang/fitness_equipments/dataset/3
3. Gym Equipment Detector Computer Vision Model 
https://universe.roboflow.com/hamzas-workspace-lgrj3/gym-equipment-detector-xzyer/dataset/3
4. Gym Equipment Computer Vision Dataset
https://universe.roboflow.com/xiaoshis-workspace/gym-equipment-t6kck/dataset/1
5. Gym Computer Vision Model
[https://universe.roboflow.com/myworkspace-czk94/gym-equipment-qkvbl/dataset/1](https://universe.roboflow.com/data-science-afe7n/gym-5mrjy/dataset/1)
6. 
  
7.  LA Fitness Machines Computer Vision Model
https://universe.roboflow.com/gymlens-for-la-fitness-mvp-6lhrg/la-fitness-machines-jrgny/dataset/2

# 2. 파이프라인

### $\color{blue}{\text{Phase 0: 데이터 정제}}$        

 ﻿$\color{blue}{\text{Step 0. 데이터셋 병합 merge dataset.py, 데이터 재분배 resplit dataset.py, 포즈 데이터 삭제 remove workout pose.py}}$

﻿1) 데이터셋 병합: 7개의 각기 다른 데이터셋은 같은 클래스에 대해 각기 다른 표기법을 사용한다.
따라서 파편화된 명칭을 재 매핑한다. 
라벨링 데이터에 포함된 class number 를 합쳐진 클래스에 따라 재할당 한다.
중복되는 이미지 명을 대비해 데이터셋에 따라 데이터셋 번호를 붙여준다.
경로 탐색 및 출력 경로에 대해 지정해준다.

2) 데이터 재배치: 7개의 각기 다른 데이터셋은 train, test, valid 분배 비율이 각기 다르다. 병합된 데이터셋을 확인해 8:1:1의 비율로 다시 재배치한다. 
이때 images와 labels가 같이 이동되도록 List 형태의 가상 풀로 통합한다.
랜덤 셔플 및 인덱스 분할로 재배치한다.

3) 포즈 데이터 삭제: 이미지 데이터셋을 roboflow 에서 가져올때 여러 데이터셋을 가져오다 보니 장비 이미지가 아닌 포즈 이미지 데이터가 일부 섞이는 오염이 발생.
검은 패딩(letterbox) — 좌우 가장자리 30px의 평균 밝기가 15 미만 (세로 영상에서 추출된 프레임의 특징)
모든 bbox가 소형 — 이미지 대비 면적 5% 미만 (기구가 아니라 손에 쥔 도구 수준)
두 기준으로 삭제 진행

$\color{blue}{\text{Step 0-1. 극소 클래스 4개 제거}}$

﻿아래 4개 클래스는 train 이미지 수가 너무 적어 어떤 증강 기법으로도 유의미한 학습이 불가능합니다. data.yaml에서 제거하고, 해당 라벨 파일에서 관련 바운딩 박스 행을 삭제합니다.

<img width="470" height="160" alt="image" src="https://github.com/user-attachments/assets/df5e538f-3a13-4c91-a2e9-32be5e7d5102" />


$\color{blue}{\text{﻿﻿﻿Step 0-2. data.yaml 수정 (이후 클래스 추가 제거로 인해 추가 수정)}}$

 nc: 37 → nc: 33으로 변경하고, names 리스트에서 4개 클래스를 제거합니다. 이때 클래스 인덱스가 변경되므로, 모든 라벨(.txt) 파일의 클래스 번호를 새 인덱스에 맞게 리매핑해야 합니다.

$\color{blue}{\text{step 0-3. 희소 클래스 오버샘플링}}$

﻿300장 미만인 클래스에 대해 이미지와 라벨 파일을 함께 복사하여 학습 폴더에 추가합니다. 복사된 파일명에 접미사(예: _aug1, _aug2)를 붙여 원본과 구분합니다.
  
<img width="425" height="205" alt="image" src="https://github.com/user-attachments/assets/dd3beaa1-17c0-49a6-8504-c7ceffb23895" />

﻿오버샘플링 후 불균형 비율: 4,368:294 = 약 15:1 (정리 전 2,184:1 대비 대폭 개선)

$\color{blue}{\text{step ﻿0-4. 학습 전 추가 성능 개선 작업}}$

﻿데이터 양을 늘리거나 품질을 높이는 다음 작업들은 학습 전에 수행할수록 모델 성능에 직접적인 영향을 줍니다.

$\color{blue}{\text{﻿(1) Albumentations 기반 오프라인 증강}}$
단순 복사 대신, Albumentations 라이브러리로 희소 클래스 이미지에 다양한 변환을 적용하여 실질적으로 다른 학습 샘플을 생성합니다.
RandomBrightnessContrast: 헬스장의 다양한 조명 환경 시뮬레이션
HorizontalFlip: 좌우 반전 (기구 대부분 좌우 대칭)
ShiftScaleRotate: 촬영 각도 변화 시뮬레이션 (±15° 이내)
GaussNoise + MotionBlur: 카메라 흔들림/노이즈 시뮬레이션
YOLO 라벨 좌표도 함께 변환해야 하므로, Albumentations의 BboxParams(format='yolo') 설정을 반드시 사용하세요.

$\color{blue}{\text{(2) 라벨 품질 검증 (Sanity Check)}}$
7개 소스 데이터셋이 병합된 상태이므로 라벨링 기준이 불일치할 수 있습니다. 학습 전에 다음을 확인합니다.
bbox 좌표 범위 검증: x, y, w, h 값이 모두 0-1 범위 내인지 확인
빈 라벨 파일 검출: 이미지는 있으나 라벨 파일이 비어 있는 케이스 확인
클래스 ID 유효성: 리매핑 후 모든 ID가 0-32 범위 내인지 확인
시각적 스팟체크: 각 클래스별 랜덤 5-10장을 뽑아 bbox를 이미지 위에 그려 육안으로 확인

$\color{blue}{\text{(3) 배경(Negative) 이미지 추가}}$
현재 데이터셋에는 배경 이미지(기구가 없는 헬스장 사진)가 없습니다. 배경 이미지를 전체의 1-3%(약 300-900장) 추가하면 False Positive를 줄이는 데 효과적입니다. 라벨 파일은 빈 파일(.txt)로 생성합니다.

$\color{blue}{\text{(4) 앵커 프리 확인 (YOLO26 전용)}}$
YOLO26은 앵커 프리(anchor-free) 아키텍처이므로 별도의 앵커 분석이나 autoanchor 설정이 필요 없습니다. 기존 YOLOv5에서 사용하던 앵커 관련 작업은 건너뛰어도 됩니다.

### $\color{blue}{\text{Phase 1: 심층 EDA}}$

﻿Phase 0의 데이터 정제가 완료된 후, 정제된 데이터셋 기준으로 최종 EDA를 수행합니다.

$\color{blue}{\text{﻿Step 1-1. 정제 후 데이터 검증 EDA}}$

﻿1. 클래스별 최종 분포 확인: 33개 클래스의 train/valid/test 분포가 비례적인지 확인
2. 오버샘플링 반영 확인: 복사/증강된 이미지가 정상적으로 반영되었는지 확인
3. BBox 크기 분포 분석: 소형 객체(bbox 면적 < 32² 픽셀) 비율을 파악하여 YOLO26의 STAL 손실 함수 효과를 예측
4. 데이터셋 간 중복 검사: 7개 소스 데이터셋 간 이미지 해시를 비교하여 train-valid-test 누수(leakage)가 없는지 확인

- 데이터셋 간 중복 검사 결과 
크로스 스플릿 중복: 71건
40×640 리사이징 시 파일 헤더가 유사해져서 발생하는 오탐인지 확인 check_duplicates.py
  
![dup_002_train_vs_train_vs_test](https://github.com/user-attachments/assets/a7f988a3-dfe3-402a-8b1a-8cb5f4fbd010)

결과 실제 중복 확인 이후 삭제

$\color{blue}{\text{﻿Step 1-2. ﻿클래스 정리 remove classes.py, data.yaml 수정}}$

﻿"Foam_Roller", "Yoga_Mat", "Gym_Ball", "Punching_Bag", "Dumbbell_Rack"
이 5개의 클래스의 경우 본래 목표인 운동 기구를 사용하는 운동법, 주의사항, 운동 영상 링크 제공에 부적합하다. 
스트레칭용도이거나 단일 동작 목적이라서 직관적이거나 운동 기구가 아니기 때문이다. 
제거된 5개 클래스의 이미지 데이터는 비교용의 배경 데이터로 사용한다. 

$\color{blue}{\text{﻿Step 1-3. ﻿CSV파일 제작 create eda csv.py}}$

﻿EDA 이후 최종 정리된 데이터셋에 대해 csv를 제작한다. 
이를 통해 데이터셋이 위 증강, 오버샘플링, 라벨링 등이 제대로 적용되었는지 확인한다. 이후 결과 분석 단계에서 비교 용도로 사용가능하다.

### $\color{blue}{\text{Phase 2: ﻿YOLO26 모델 학습 및 성능 튜닝}}$

$\color{blue}{\text{﻿Step 2-1. 모델 선택 및 기본 설정}}$

<img width="425" height="180" alt="image" src="https://github.com/user-attachments/assets/a9442770-f147-45eb-ae10-71d34601e024" />

﻿> time = 1.8 (108분)으로 epoch 15진행 phase 5의 서비스 제작 체크로 epoch 15로도 좋은 성능이 나와 베이스라인은 이대로 진행.

$\color{blue}{\text{﻿Step 2-2. ﻿증강(Augmentation) 설정}}$

<img width="425" height="190" alt="image" src="https://github.com/user-attachments/assets/ffe7578f-6a26-4291-bae4-43b19ea401ad" />

$\color{blue}{\text{﻿Step 2-3. ﻿﻿YOLO26 아키텍처 특성 반영}}$

﻿- NMS-Free end-to-end 추론: YOLO26은 NMS 후처리가 제거되었습니다. conf threshold와 max_det 값을 검증 단계에서 면밀히 조정하세요.
- STAL (Small-Target-Aware Label Assignment): 소형 객체 탐지 성능이 내장 개선되어 있으므로, 별도의 소형 객체 대응 전략은 불필요합니다.
- ProgLoss (Progressive Loss Balancing): 학습 안정화가 내장되어 있어, 학습 초반 불안정 시 별도 스케줄링 조정이 덜 필요합니다.
- DFL 제거: CoreML/TFLite 변환이 간소화되어 Phase 5 iOS 배포에 직접적 이점이 있습니다.

### $\color{blue}{\text{Phase 3: ﻿성능 검증 및 테스트}}$

$\color{blue}{\text{﻿Step 3-1. ﻿전체 성능 지표 확인}}$
- ﻿mAP@50, mAP@50:95: 전체 클래스 기준 평균 정밀도 확인
- Precision / Recall: 클래스별 균형 확인
- Confusion Matrix: 클래스 간 혼동 패턴 분석 (유사 기구 간 오분류 확인)

$\color{blue}{\text{﻿Step 3-2. ﻿﻿희소 클래스 개별 검증 (팩트 체크)}}$
﻿전체 mAP에 가려질 수 있는 하위 클래스를 별도로 검증합니다. 특히 오버샘플링을 적용한 7개 클래스의 개별 AP를 확인합니다.

<img width="425" height="155" alt="image" src="https://github.com/user-attachments/assets/a46c97f7-59a9-4efa-b349-90c43f8e3d51" />

$\color{blue}{\text{﻿Step 3-3. ﻿﻿﻿유사 기구 혼동 분석}}$

﻿헬스장 기구 특성상 형태가 유사한 클래스 쌍이 존재합니다. Confusion Matrix에서 다음 쌍의 교차 오분류율을 특별히 확인합니다.
- Chest_Press ↔ Shoulder_Press (앉아서 미는 동작 기구)
- Leg_Curl ↔ Leg_Extension (다리 운동 기구)
- Elliptical ↔ Stationary_Bike (유산소 기구)
- Aerobic_Stepper ↔ Plyo_Box (박스형 기구)
- Kettlebell ↔ Dumbbell (손잡이형 중량 기구)

$\color{blue}{\text{﻿Step 3-4. ﻿﻿﻿﻿test 이미지 추론 시각화}}$


### $\color{blue}{\text{Phase 4: ﻿Gradio 웹 프로토타입 서비스}}$

$\color{blue}{\text{﻿Step 4-1. ﻿메타데이터 (JSON) 고도화 (equipment guide_v4.json)}}$

﻿> 학습이 완료된 타겟 클래스 28개(유산소 4종, 근력 24종)에 대한 전문가 수준의 JSON 데이터베이스를 구축 및 연동합니다.
﻿- 정보 구성: 기구 정밀 세팅법(machine_setup), 통증 발생 시 대처법(pain_management), 세부 운동 모드별 호흡법(breathing) 및 가이드.
- 유튜브 맞춤 연동: 각 기구의 세부 운동 모드(exercise_modes) 특성에 맞춰, 조회수가 높고 길이가 짧은 한국어 기반 쇼츠(1분 요약) 검색 다이내믹 링크(youtube_url)를 46개 매핑 완료.
- 시각적 가이드 매핑: 근력 운동의 수축(concentric) 및 이완(eccentric) 동작, 유산소(cardio) 동작을 보여주는 에셋 파일명을 JSON 구조에 할당.

$\color{blue}{\text{﻿Step 4-2. ﻿﻿﻿﻿시각적 가이드 에셋 전처리 파이프라인 (resize images.py)}}$

> ﻿Gradio UI에서 이미지가 깨지거나 레이아웃이 틀어지는 것을 방지하기 위한 전처리 스크립트를 운용합니다.
- 비율 유지 리사이징: 수집된 원본 자세 이미지의 가로세로 비율(Aspect Ratio)을 파괴하지 않고 가장 긴 변을 640px에 맞춤.
- 레터박싱(Letterboxing) 처리: 남는 여백 공간은 YOLO 학습 기본 패딩 색상(RGB: 114, 114, 114)으로 채워 640x640 정방형 이미지를 자동 생성하여 assets/images/ 디렉토리에 저장.<img width="851" height="130" alt="image" src="https://github.com/user-attachments/assets/f356b563-570b-4093-8c07-90aa794ec761" />

$\color{blue}{\text{﻿Step 4-3. ﻿﻿﻿﻿﻿Gradio 프레임워크 기반 웹 UI 추론 구현 (app.py)}}$

> ﻿YOLO 모델과 JSON 메타데이터를 결합하여 사용자 피드백을 실시간으로 제공하는 인터페이스를 구축합니다.<img width="937" height="51" alt="image" src="https://github.com/user-attachments/assets/7ce9c99c-86d8-4ffb-bd08-20bebca3f51d" />
- ﻿입력 인터페이스: 웹캠 촬영 및 로컬 이미지 업로드 동시 지원.
- YOLO 추론 연동: 학습된 가중치(best.pt)를 로드하여 이미지 내 기구를 NMS-Free End-to-End 방식으로 탐지하고 바운딩 박스를 시각화.
- 동적 UI 렌더링: 탐지된 객체(가장 Confidence가 높은 기구 1개)의 영문 클래스명을 키(Key)값으로 사용하여, equipment_guide_v4.json에서 해당 데이터를 파싱. 화면 우측에 아코디언(Accordion) 메뉴 형태로 세팅법, 통증 관리, 갤러리 이미지, 유튜브 링크를 렌더링.


### $\color{blue}{\text{Phase 5: ﻿ios 네이티브 앱 구축}}$

$\color{blue}{\text{﻿Step 5-1. ﻿모델 배포 전략 수립}}$

- ﻿YOLO26의 DFL 제거 + NMS-Free 아키텍처 덕분에 CoreML 변환이 이전 버전보다 간소화되었습니다. 두 가지 배포 전략을 병행 검토합니다.

<img width="425" height="190" alt="image" src="https://github.com/user-attachments/assets/50a4ecd0-172a-4309-8ade-eb2db99764bf" />

$\color{blue}{\text{﻿Step 5-2. ﻿﻿백엔드 (서버 추론 시)}}$
﻿- FastAPI 기반 REST API 서버 구축
- YOLO26 추론 엔드포인트 + 메타데이터 DB 연동
- 응답 포맷: JSON (클래스 ID, confidence, bbox, 메타데이터)

$\color{blue}{\text{﻿Step 5-3. ﻿﻿﻿프론트엔드 (iOS)}}$
﻿- Swift 또는 Flutter로 iOS 전용 UI/UX 개발
- 카메라 촬영 → 온디바이스 추론(CoreML) 또는 서버 API 호출
- 운동 가이드 화면: 기구 정보 + 운동법 + 주의사항 + 영상 링크 통합 표시

<br>

# 3. 데이터
- 데이터 전처리 이후 image 파일 이름과 label을 통해 추출한 정보

<img width="280" height="340" alt="image" src="https://github.com/user-attachments/assets/8b2bcb81-7a25-42b4-8667-160239e91861" />

<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/0b56e586-4df5-4fcc-8c11-81994e8a6d75" />

── Split 비율 체크 ──
  Train: 35761장 (82.5%)
  Valid: 3799장 (8.8%)
  Test: 3790장 (8.7%)
  
<img width="175" height="220" alt="image" src="https://github.com/user-attachments/assets/7ed86e9d-13a8-4c3b-8061-db4004eaec41" />

<img width="900" height="245" alt="image" src="https://github.com/user-attachments/assets/c3453aad-9dc5-4a2e-82ff-0c1bb9efac2d" />

# 4. 실험

## 0. baseline

|   name   | YOLO26 model | epoch | batch | imgsz | metric (mAP50) |
|:--------:|:------------:|:-----:|:-----:|:-----:|:-----------------:|
| baseline |     small    |   15  |  48  |  640  |       0.950      |


<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/b2856bac-37b3-400e-b0e6-a4ccf7bce11f" />

> 유사 기구 혼동 매트릭스 결과
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/d5337b30-0f62-462d-b0e1-9c9d5b8c02a5" />

> Test 이미지 추론 시각화 결과
<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/0a864ff9-392c-4c57-8ea9-b7b26f54e5e2" />

## 실험 1: 외부 .pt 파일 (사용한 데이터셋의 일부(7개 중 1개 데이터셋으로 학습))이 학습에 반영 되었을때 성능 변화 비교 
- 같은 주제를 선정한 정영석님의 small.pt를 학습에 활용

## 실험 2: model size & epoch 변경 (medium, nano)
- 같은 주제를 선정한 정영석님의 nano.pt, small.pt, medium.pt 를 학습에 활용

# 5. 결과

## 실험 1: 외부 .pt파일 적용 결과

## 실험 2: model size & epoch 변경 결과

# 6. 프로젝트 회고

### 어려웠던 점
- 이미지가 3만 - 4만개 정도의 적은 데이터양이지만 small 모델을 사용해도 batch 48의 경우 1개 epoch마다 7분 이상 소요되어 다양한 시도를 해보기 쉽지 않았습니다.
- roboflow에서 데이터셋을 가져왔지만 데이터 표기나 split 비율도 다르고 데이터셋별 이미지 장수가 달라 병합 후에 클래스별 이미지 장수 차이가 커 증강이후에도 성능 차이가 컸습니다.

### 보완할 점

- 본래 목적은 헬스장의 처음보는 독특한 머신의 명칭과 사용방법을 알기 위해 선정한 주제이지만 이후 찾아보니 일반적이지 않은 머신들은 대부분 특정 제조업체의 독창적인 제품이고 이 수가 매우 많으며 업체에 요청해 api를 받는게 아니면 데이터셋을 만들기가 현실적으로 어렵다는 것을 확인했습니다.
- 제공하는 자세에 대한 참고 이미지는 구글링을 통해 찾은 뒤 보정을 한 이미지로 상업적 사용이 불가능합니다. 하지만 YOLO의 Pose estimation 등을 사용해 유튜브의 자세 교육 동영상들을 학습하여 자체 데이터 생성이 가능해 보입니다.
- 온디바이스의 경우 사용자의 자세를 실시간 입력해 자세 교정 멘트를 제공할수도 있습니다.
- 장비들마다 훨씬 많은 방대한 운동법이 있으나 대표적인 운동만 추가한것으로 차후 더 많은 내용을 추가 가능합니다.




