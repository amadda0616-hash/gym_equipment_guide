# gym_equipment_guide
> 헬스장 장비 인식 후 사용설명 제공 서비스 프로젝트

# 📃 Contents

[1. 프로젝트 소개](#1-프로젝트-소개) <br>
  - [목표](#목표)
  - 
  
[2. 파이프라인](#2-파이프라인) <br>
  - [phase 0: 데이터 정제](#phase 0: 데이터 정제)
  - [phase 1: EDA + 전처리](#phase 1: EDA + 전처리)
  - [phase 2: 실험 (YOLO26 학습)](#phase 2: 실험)
  - [phase 3: 결과 (성능 검증)](#phase 3: 결과)
  - [phase 4: Gradio 웹서비스](#phase 4: Gradio 웹서비스)
  - [phase 5: ios 앱 서비스](#phase 5: ios 앱 서비스) 

[3. 실험](#3-실험) <br>
  - [baseline](#0-baseline)

[4. 결과](#4-결과) <br>

[5. 프로젝트 회고](#5-프로젝트-회고) <br>
  - [어려웠던 점](#어려웠던-점)
  - [배운 점](#배운-점)


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
