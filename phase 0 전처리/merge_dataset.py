import os
import shutil
import yaml

# 1. 경로 및 통합 클래스 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_BASE_DIR = os.path.join(BASE_DIR, 'data', 'dataset')

DATASET_FOLDERS = [
    "Fitness Equipment Recognition.v3-version-2.yolo26",
    "fitness_equipments_train.v1i.yolo26",
    "Gym Equipment Detector.v3-v3--to-train-roboflow-.yolo26",
    "gym equipment object detection.v1i.yolo26",
    "Gym Equipment.v1i.yolo26",
    "gym.v4i.yolo26",
    "LA Fitness Machines.v2-v2-gym-machines-split.yolo26"
]

CLASS_MAPPING = {
    'Ab Wheel': 'Ab_Wheel', 'AbWheel': 'Ab_Wheel',
    'Aerobic Stepper': 'Aerobic_Stepper', 'AerobicStepper': 'Aerobic_Stepper',
    'Arm Curl': 'Arm_Curl', 'ArmCurl': 'Arm_Curl', 'arm curl machine': 'Arm_Curl',
    'Assisted Chin Up-Dip': 'Assisted_Chin_Up_Dip', 'AssistedChinUpDip': 'Assisted_Chin_Up_Dip', 'chinning dipping': 'Assisted_Chin_Up_Dip', 'pullup bar': 'Assisted_Chin_Up_Dip',
    'Back Extension': 'Back_Extension', 'BackExtension': 'Back_Extension',
    'Barbell': 'Barbell', 'barbell': 'Barbell', 'Barbell Bar': 'Barbell',
    'Cable Machine': 'Cable_Machine', 'CableMachine': 'Cable_Machine', 'functional_trainer': 'Cable_Machine',
    'Chest Fly': 'Chest_Fly', 'ChestFly': 'Chest_Fly', 'chest fly machine': 'Chest_Fly',
    'Chest Press': 'Chest_Press', 'ChestPress': 'Chest_Press', 'Chest Press machine': 'Chest_Press', 'bench-press': 'Chest_Press',
    'Dumbbell': 'Dumbbell', 'dumb-bell': 'Dumbbell',
    'Gymball': 'Gym_Ball', 'exercise ball': 'Gym_Ball', 'Gym ball': 'Gym_Ball',
    'Hip Abductor': 'Hip_Abductor', 'HipAbductor': 'Hip_Abductor',
    'Kettlebell': 'Kettlebell', 'Kettle ball': 'Kettlebell',
    'Lat Pulldown': 'Lat_Pulldown', 'LatPulldown': 'Lat_Pulldown', 'Lat Pull Down': 'Lat_Pulldown', 'lat-pull-down-machine': 'Lat_Pulldown', 'lat_pulldown': 'Lat_Pulldown',
    'Leg Extension': 'Leg_Extension', 'LegExtension': 'Leg_Extension', 'leg extension': 'Leg_Extension', 'leg_extension': 'Leg_Extension',
    'Leg Press': 'Leg_Press', 'LegPress': 'Leg_Press', 'leg press': 'Leg_Press', 'leg-press': 'Leg_Press', 'leg_press': 'Leg_Press',
    'Lying Leg Curl': 'Leg_Curl', 'LyingLegCurl': 'Leg_Curl', 'reg curl machine': 'Leg_Curl', 'seated-leg-ext.-curl': 'Leg_Curl',
    'Person': 'Person',
    'Punching Bag': 'Punching_Bag', 'PunchingBag': 'Punching_Bag', 'Bananabag': 'Punching_Bag',
    'Shoulder Press': 'Shoulder_Press', 'ShoulderPress': 'Shoulder_Press', 'shoulder press machine': 'Shoulder_Press', 'shoulder_press': 'Shoulder_Press',
    'Smith Machine': 'Smith_Machine', 'SmithMachine': 'Smith_Machine', 'smith machine': 'Smith_Machine',
    'Stationary Bike': 'Stationary_Bike', 'StationaryBike': 'Stationary_Bike',
    'T-Bar Row': 'T_Bar_Row', 'TBarRow': 'T_Bar_Row',
    'Treadmill': 'Treadmill', 'treadmill': 'Treadmill',
    'Foam Roller': 'Foam_Roller',
    'Jump Rope': 'Jump_Rope',
    'Resistance Band': 'Resistance_Band',
    'Weight Plates': 'Weight_Plates',
    'Yoga matt': 'Yoga_Mat', 'gym-mat': 'Yoga_Mat',
    'Seated Cable Rows': 'Seated_Cable_Row', 'seated_row': 'Seated_Cable_Row',
    'lateral raises machine': 'Lateral_Raise',
    'seated dip machine': 'Seated_Dip',
    'Clubbell': 'Clubbell',
    'Medicineball': 'Medicine_Ball',
    'box-stair': 'Plyo_Box',
    'dumbbell rack': 'Dumbbell_Rack',
    'eliptical cycle': 'Elliptical'
}

MASTER_CLASSES = sorted(list(set(CLASS_MAPPING.values())))
MASTER_CLASS_TO_ID = {cls_name: idx for idx, cls_name in enumerate(MASTER_CLASSES)}
TOTAL_CLASSES = len(MASTER_CLASSES)

SPLITS = ['train', 'test', 'valid']

# 타겟 폴더 초기화 생성
for split in SPLITS:
    os.makedirs(os.path.join(TARGET_BASE_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_BASE_DIR, split, 'labels'), exist_ok=True)

def find_image_file(image_dir, base_name):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def process_datasets():
    total_images_copied = 0
    total_folders = len(DATASET_FOLDERS)

    for dataset_idx, folder_name in enumerate(DATASET_FOLDERS, start=1):
        print(f"\n[{dataset_idx}/{total_folders}] '{folder_name}' 데이터셋 병합 중...")
        src_dir = os.path.join(BASE_DIR, folder_name)
        
        if not os.path.exists(src_dir):
            print(f"  -> [오류] 경로 탐색 실패: {src_dir}")
            continue

        yaml_files = [f for f in os.listdir(src_dir) if f.endswith('.yaml')]
        if not yaml_files:
            continue
        
        with open(os.path.join(src_dir, yaml_files[0]), 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            original_names = data_yaml['names']
        
        dataset_copied = 0
        for split in SPLITS:
            src_split_dir = os.path.join(src_dir, split)
            src_img_dir = os.path.join(src_split_dir, 'images')
            src_lbl_dir = os.path.join(src_split_dir, 'labels')
            
            if not os.path.exists(src_lbl_dir) or not os.path.exists(src_img_dir):
                continue

            label_files = [f for f in os.listdir(src_lbl_dir) if f.endswith('.txt')]
            
            for label_file in label_files:
                base_name = os.path.splitext(label_file)[0]
                img_path = find_image_file(src_img_dir, base_name)
                lbl_path = os.path.join(src_lbl_dir, label_file)
                
                if not img_path:
                    continue
                
                new_label_lines = []
                with open(lbl_path, 'r', encoding='utf-8') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        orig_cls_id = int(parts[0])
                        if orig_cls_id >= len(original_names):
                            continue
                            
                        orig_cls_name = original_names[orig_cls_id]
                        
                        if orig_cls_name in CLASS_MAPPING:
                            master_name = CLASS_MAPPING[orig_cls_name]
                            new_cls_id = MASTER_CLASS_TO_ID[master_name]
                            new_line = f"{new_cls_id} {' '.join(parts[1:])}\n"
                            new_label_lines.append(new_line)
                
                # 라벨 내용이 유효한 경우에만 복사 진행 (빈 라벨 방지)
                if new_label_lines:
                    new_base_name = f"d{dataset_idx}_{base_name}"
                    dest_img_path = os.path.join(TARGET_BASE_DIR, split, 'images', new_base_name + os.path.splitext(img_path)[1])
                    dest_lbl_path = os.path.join(TARGET_BASE_DIR, split, 'labels', new_base_name + '.txt')
                    
                    shutil.copy2(img_path, dest_img_path)
                    with open(dest_lbl_path, 'w', encoding='utf-8') as df:
                        df.writelines(new_label_lines)
                    
                    total_images_copied += 1
                    dataset_copied += 1
                    
                    if dataset_copied % 500 == 0:
                        print(f"  -> {dataset_copied}장 처리 완료...")
                        
        print(f"  -> '{folder_name}' 완료 (총 {dataset_copied}장 복사됨)")

def create_master_yaml():
    yaml_content = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': TOTAL_CLASSES,
        'names': MASTER_CLASSES
    }
    yaml_path = os.path.join(TARGET_BASE_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

# --- 신규 추가: 데이터 무결성 검증 함수 ---
def verify_dataset():
    print("\n=====================================")
    print("데이터 무결성 검증(Verification) 시작...")
    print("=====================================")
    
    total_issues = 0
    
    for split in SPLITS:
        img_dir = os.path.join(TARGET_BASE_DIR, split, 'images')
        lbl_dir = os.path.join(TARGET_BASE_DIR, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
            
        images = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))]
        labels = [os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')]
        
        set_images = set(images)
        set_labels = set(labels)
        
        # 1. 일대일 매칭 검사
        img_without_lbl = set_images - set_labels
        lbl_without_img = set_labels - set_images
        
        print(f"\n[{split.upper()} 폴더 검증 결과]")
        print(f" - 총 이미지 수: {len(images)}장")
        print(f" - 총 라벨 수: {len(labels)}개")
        
        if img_without_lbl:
            print(f"  [경고] 라벨이 없는 이미지: {len(img_without_lbl)}건")
            total_issues += len(img_without_lbl)
        if lbl_without_img:
            print(f"  [경고] 이미지가 없는 라벨: {len(lbl_without_img)}건")
            total_issues += len(lbl_without_img)
            
        # 2. 라벨 데이터 오염 검사 (범위 초과 클래스 ID 확인)
        invalid_labels = 0
        empty_labels = 0
        for lbl_name in labels:
            lbl_path = os.path.join(lbl_dir, lbl_name + '.txt')
            if os.path.getsize(lbl_path) == 0:
                empty_labels += 1
                continue
                
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < 0 or cls_id >= TOTAL_CLASSES:
                            invalid_labels += 1
                            break
                            
        if empty_labels > 0:
            print(f"  [경고] 내용이 텅 빈 라벨 파일: {empty_labels}건")
            total_issues += empty_labels
        if invalid_labels > 0:
            print(f"  [경고] 존재하지 않는 클래스 ID를 포함한 라벨: {invalid_labels}건")
            total_issues += invalid_labels
            
        if not img_without_lbl and not lbl_without_img and empty_labels == 0 and invalid_labels == 0:
            print("  -> 상태: 완벽함 (결함 없음)")

    print("\n=====================================")
    if total_issues == 0:
        print("최종 팩트 체크: 모든 데이터가 1:1로 완벽하게 매칭되었으며, 오염된 라벨이 없습니다. 학습 준비가 완료되었습니다.")
    else:
        print(f"최종 팩트 체크: 총 {total_issues}건의 데이터 결함이 발견되었습니다. 위 경고를 확인하여 해당 파일을 수동으로 삭제하거나 수정해야 합니다.")
    print("=====================================")

if __name__ == "__main__":
    print("데이터 병합을 시작합니다...\n")
    process_datasets()
    create_master_yaml()
    verify_dataset()