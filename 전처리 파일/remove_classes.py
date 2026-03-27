import os
import yaml
from pathlib import Path

# 1. 경로 및 삭제 대상 설정
DATASET_ROOT = Path(".")
DATA_YAML = DATASET_ROOT / "data.yaml"
TARGET_CLASSES_TO_REMOVE = [
    "Foam_Roller", 
    "Yoga_Mat", 
    "Gym_Ball", 
    "Punching_Bag", 
    "Dumbbell_Rack"
]

def remove_classes_and_reindex():
    if not DATA_YAML.exists():
        print(f"[오류] {DATA_YAML} 파일이 존재하지 않습니다.")
        return

    # 기존 yaml 로드
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    original_names = data_cfg.get("names", [])
    
    # 2. 삭제 대상 ID 식별
    ids_to_remove = set()
    for idx, name in enumerate(original_names):
        if name in TARGET_CLASSES_TO_REMOVE:
            ids_to_remove.add(idx)

    print(f"삭제 예정 클래스 수: {len(ids_to_remove)}개")
    for idx in ids_to_remove:
        print(f" - [ID: {idx}] {original_names[idx]}")

    # 3. 새 클래스 목록 및 ID 매핑 테이블 생성
    new_names = []
    old_to_new_mapping = {}
    new_idx = 0

    for old_idx, name in enumerate(original_names):
        if old_idx in ids_to_remove:
            old_to_new_mapping[old_idx] = None  # 삭제 대상
        else:
            new_names.append(name)
            old_to_new_mapping[old_idx] = new_idx
            new_idx += 1

    print(f"\n유지되는 새로운 클래스 수: {len(new_names)}개 (0 ~ {len(new_names)-1}번)")
    print("\n라벨 파일 내부 바운딩 박스 삭제 및 ID 재정렬을 시작합니다...")

    # 4. 모든 라벨 파일 순회하며 수정
    splits = ["train", "valid", "test"]
    modified_files_count = 0
    removed_lines_count = 0

    for split in splits:
        label_dir = DATASET_ROOT / split / "labels"
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            file_modified = False

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                # 팩트 체크: '30.0' 같은 실수형 텍스트 에러 방어용 이중 캐스팅
                old_id = int(float(parts[0]))
                
                # 라벨 유지/삭제/수정 판단
                if old_id in ids_to_remove:
                    removed_lines_count += 1
                    file_modified = True
                elif old_id in old_to_new_mapping:
                    new_id = old_to_new_mapping[old_id]
                    if old_id != new_id:
                        file_modified = True
                    # 새로운 ID로 덮어쓰기
                    new_line = f"{new_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)

            # 변경사항이 있는 파일만 덮어쓰기 저장
            if file_modified:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                modified_files_count += 1

    # 5. data.yaml 백업 및 갱신
    backup_yaml = DATASET_ROOT / "data_backup.yaml"
    if not backup_yaml.exists():
        with open(backup_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data_cfg, f, sort_keys=False)

    data_cfg["names"] = new_names
    data_cfg["nc"] = len(new_names)

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, sort_keys=False)

    print("\n==================================================")
    print("모든 정제 작업이 성공적으로 완료되었습니다!")
    print(f" - 기존 data.yaml이 'data_backup.yaml'로 백업되었습니다.")
    print(f" - ID가 수정되거나 줄이 삭제된 라벨 파일 수: {modified_files_count}개")
    print(f" - 완전히 도려낸 바운딩 박스(줄) 수: {removed_lines_count}개")
    print("==================================================")

if __name__ == "__main__":
    remove_classes_and_reindex()