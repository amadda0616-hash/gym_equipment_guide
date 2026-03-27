import os
import shutil
import random
import pandas as pd
from collections import Counter

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'dataset')
BACKUP_DIR = os.path.join(BASE_DIR, 'data', 'dataset_old')
CSV_PATH = os.path.join(BASE_DIR, 'create_eda.csv')
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, 'create_eda_v2.csv')

def smart_resplit():
    # 팩트 체크: 필요 파일 및 폴더 존재 여부 확인
    if not os.path.exists(CSV_PATH):
        print(f"[오류] 기준이 될 CSV 파일이 없습니다: {CSV_PATH}")
        return
    if not os.path.exists(DATASET_DIR):
        print(f"[오류] 원본 데이터셋 폴더가 없습니다: {DATASET_DIR}")
        return

    print("CSV 데이터를 분석하여 스마트 재분배(8:1:1) 계획을 수립합니다...")
    df = pd.read_csv(CSV_PATH)
    total_images = len(df)

    # 2. 희소 클래스 파악 (출현 빈도 50회 미만)
    all_classes = []
    for classes_str in df['Classes_Included'].dropna():
        if classes_str != 'Background_Only':
            all_classes.extend([c.strip() for c in classes_str.split(',')])
            
    class_counts = Counter(all_classes)
    rare_classes = {cls for cls, count in class_counts.items() if count < 50}
    
    print(f" -> 보호 대상 희소 클래스 (100% Train 배정): {rare_classes}")

    # 희소 클래스가 포함된 이미지 판별 함수
    def has_rare_class(classes_str):
        if pd.isna(classes_str) or classes_str == 'Background_Only':
            return False
        cls_list = [c.strip() for c in classes_str.split(',')]
        return any(c in rare_classes for c in cls_list)

    # 3. 새로운 분할 인덱스(Split) 계산
    # 희소 클래스가 있는 행은 무조건 'train'으로 고정
    rare_mask = df['Classes_Included'].apply(has_rare_class)
    forced_train_count = rare_mask.sum()
    print(f" -> 희소 클래스 보호를 위해 강제 Train 배정된 이미지: {forced_train_count}장")

    # 목표 수량 계산 (8:1:1)
    target_train = int(total_images * 0.8)
    target_valid = int(total_images * 0.1)
    
    # 남은 배정 가능 슬롯 계산
    remaining_train_needed = max(0, target_train - forced_train_count)
    
    # 배정되지 않은 나머지 인덱스 무작위 섞기
    remaining_indices = df[~rare_mask].index.tolist()
    random.seed(42) # 재현성을 위한 시드 고정
    random.shuffle(remaining_indices)

    # 남은 슬롯에 맞게 인덱스 분배
    train_idx = remaining_indices[:remaining_train_needed]
    valid_idx = remaining_indices[remaining_train_needed : remaining_train_needed + target_valid]
    test_idx = remaining_indices[remaining_train_needed + target_valid :]

    # DataFrame에 'New_Split' 컬럼 적용
    df['New_Split'] = None
    df.loc[rare_mask, 'New_Split'] = 'train'
    df.loc[train_idx, 'New_Split'] = 'train'
    df.loc[valid_idx, 'New_Split'] = 'valid'
    df.loc[test_idx, 'New_Split'] = 'test'

    print("\n물리적 파일 재배치를 시작합니다. (안전을 위해 기존 폴더 백업 진행)")

    # 4. 물리적 파일 이동 준비 (백업)
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    os.rename(DATASET_DIR, BACKUP_DIR)
    
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(DATASET_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, 'labels'), exist_ok=True)

    # 5. 파일 이동 실행 (CSV의 Old Split 경로에서 New Split 경로로)
    moved_count = 0
    for index, row in df.iterrows():
        img_name = row['Image_Name']
        old_split = row['Split']
        new_split = row['New_Split']
        
        base_name = os.path.splitext(img_name)[0]
        
        old_img_path = os.path.join(BACKUP_DIR, old_split, 'images', img_name)
        old_lbl_path = os.path.join(BACKUP_DIR, old_split, 'labels', base_name + '.txt')
        
        new_img_path = os.path.join(DATASET_DIR, new_split, 'images', img_name)
        new_lbl_path = os.path.join(DATASET_DIR, new_split, 'labels', base_name + '.txt')

        # 이미지 이동
        if os.path.exists(old_img_path):
            shutil.move(old_img_path, new_img_path)
            # 라벨 이동 (라벨이 없는 배경 이미지일 경우 무시)
            if os.path.exists(old_lbl_path):
                shutil.move(old_lbl_path, new_lbl_path)
            moved_count += 1
            
        if moved_count % 5000 == 0:
            print(f" -> {moved_count}장 이동 완료...")

    # data.yaml 복구
    old_yaml = os.path.join(BACKUP_DIR, 'data.yaml')
    if os.path.exists(old_yaml):
        shutil.copy2(old_yaml, os.path.join(DATASET_DIR, 'data.yaml'))

    # 6. V2 CSV 저장
    # 기존 Split 컬럼을 갱신하고 New_Split 임시 컬럼 삭제
    df['Split'] = df['New_Split']
    df = df.drop(columns=['New_Split'])
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print("\n=====================================")
    print("데이터 재분배 및 V2 CSV 생성 완료!")
    print(f"- Train: {(df['Split'] == 'train').sum()}장")
    print(f"- Valid: {(df['Split'] == 'valid').sum()}장")
    print(f"- Test:  {(df['Split'] == 'test').sum()}장")
    print(f"- 생성된 파일: {OUTPUT_CSV_PATH}")
    print("=====================================")

if __name__ == "__main__":
    smart_resplit()