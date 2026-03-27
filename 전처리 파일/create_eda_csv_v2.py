import pandas as pd
import yaml
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# 1. 경로 설정 (전략 반영: 상대 경로로 OS 충돌 방지)
DATASET_DIR = Path(".")
YAML_PATH = DATASET_DIR / 'data.yaml'
OUTPUT_CSV = DATASET_DIR / 'create_eda_v3.csv'

def generate_eda_csv():
    # 데이터셋 및 yaml 존재 여부 팩트 체크
    if not YAML_PATH.exists():
        print(f"[오류] data.yaml 파일을 찾을 수 없습니다: {YAML_PATH.resolve()}")
        return

    # 마스터 클래스 이름 불러오기 (28개로 갱신된 yaml 기준)
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml.get('names', [])

    splits = ['train', 'valid', 'test']
    csv_data = []

    print(f"현재 인식된 클래스 수: {len(class_names)}개")
    print("이미지 및 라벨 데이터를 스캔하여 CSV 작성을 시작합니다...")

    for split in splits:
        img_dir = DATASET_DIR / split / 'images'
        lbl_dir = DATASET_DIR / split / 'labels'

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        # 이미지 폴더 내 파일 순회
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            img_name = img_path.name
            base_name = img_path.stem
            lbl_path = lbl_dir / f"{base_name}.txt"

            # 1. Source 추출 (파일명 파싱)
            source_dataset = "Unknown"
            if img_name.startswith('d') and '_' in img_name:
                possible_source = img_name.split('_')[0]
                if possible_source[1:].isdigit():
                    source_dataset = f"Dataset_{possible_source[1:]}"

            # 2. 이미지 해상도 추출 (손상 방어)
            width, height = 0, 0
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except UnidentifiedImageError:
                print(f"[경고] 손상된 이미지 파일: {img_name} (해상도를 0으로 기록합니다)")

            # 3. 라벨 분석 (포함된 클래스 및 바운딩 박스 개수)
            total_bboxes = 0
            included_classes_set = set()

            # 빈 파일이 아니며 라벨이 존재하는 경우에만 읽기
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                with open(lbl_path, 'r', encoding='utf-8') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            total_bboxes += 1
                            # 팩트 체크 방어 로직: '30.0' 형태의 에러 방지 이중 캐스팅
                            cls_id = int(float(parts[0]))
                            if cls_id < len(class_names):
                                included_classes_set.add(class_names[cls_id])
                            else:
                                included_classes_set.add(f"Unknown_ID_{cls_id}")

            # 리스트를 문자열로 변환
            classes_str = ", ".join(sorted(list(included_classes_set)))

            # 4. 데이터 행 추가
            csv_data.append({
                'Image_Name': img_name,
                'Split': split,
                'Source_Dataset': source_dataset,
                'Classes_Included': classes_str,
                'Total_BBoxes': total_bboxes,
                'Width': width,
                'Height': height,
                'Is_Original': "_os" not in img_name and "_alb" not in img_name
            })

    # DataFrame 생성 및 저장
    df = pd.DataFrame(csv_data)
    
    # 5. 배경 이미지 처리
    # 이전 단계에서 라벨이 완전히 지워진 5개 클래스 단독 이미지는 여기서 'Background_Only'로 라벨링됩니다.
    df['Classes_Included'] = df['Classes_Included'].replace('', 'Background_Only')

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\n=====================================")
    print(f"작업 완료! '{OUTPUT_CSV.name}' 파일이 생성되었습니다.")
    print(f"총 분석된 이미지 수: {len(df)}장")
    print(f"배경(Background_Only) 전환 이미지 수: {(df['Classes_Included'] == 'Background_Only').sum()}장")
    print("=====================================")

if __name__ == "__main__":
    generate_eda_csv()