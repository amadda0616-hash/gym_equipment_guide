"""
Step 0-2. data.yaml 수정 + 클래스 인덱스 리매핑
──────────────────────────────────────────────────
- nc: 37 → nc: 33
- names 리스트에서 4개 클래스 제거
- 모든 라벨(.txt) 파일의 클래스 ID를 새 인덱스로 리매핑
- 리매핑 전후 검증 리포트 출력

⚠️ 반드시 step_0_1_remove_classes.py 실행 후에 사용하세요.

사용법:
  python step_0_2_remap_classes.py --dataset_root /path/to/dataset
"""

import os
import argparse
import yaml
from pathlib import Path
from collections import Counter


# ── 매핑 정의 ──
ORIGINAL_CLASSES = [
    "Ab_Wheel", "Aerobic_Stepper", "Arm_Curl", "Assisted_Chin_Up_Dip",
    "Back_Extension", "Barbell", "Cable_Machine", "Chest_Fly", "Chest_Press",
    "Clubbell", "Dumbbell", "Dumbbell_Rack", "Elliptical", "Foam_Roller",
    "Gym_Ball", "Hip_Abductor", "Jump_Rope", "Kettlebell", "Lat_Pulldown",
    "Lateral_Raise", "Leg_Curl", "Leg_Extension", "Leg_Press", "Medicine_Ball",
    "Person", "Plyo_Box", "Punching_Bag", "Resistance_Band",
    "Seated_Cable_Row", "Seated_Dip", "Shoulder_Press", "Smith_Machine",
    "Stationary_Bike", "T_Bar_Row", "Treadmill", "Weight_Plates", "Yoga_Mat",
]

REMOVE_CLASSES = {"Jump_Rope", "Person", "Resistance_Band", "Weight_Plates"}

NEW_CLASSES = [c for c in ORIGINAL_CLASSES if c not in REMOVE_CLASSES]

# old_id → new_id 매핑 테이블
OLD_TO_NEW = {}
for old_idx, cls_name in enumerate(ORIGINAL_CLASSES):
    if cls_name not in REMOVE_CLASSES:
        OLD_TO_NEW[old_idx] = NEW_CLASSES.index(cls_name)


def remap_label_file(label_path: Path, dry_run: bool = False) -> dict:
    """
    라벨 파일의 클래스 ID를 새 인덱스로 리매핑합니다.

    Returns:
        dict: {'remapped': int, 'skipped_remove': int, 'errors': list}
    """
    result = {"remapped": 0, "skipped_remove": 0, "errors": []}

    if not label_path.exists():
        return result

    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        try:
            old_id = int(parts[0])
        except ValueError:
            result["errors"].append(f"{label_path.name}:{line_num} 잘못된 클래스 ID: '{parts[0]}'")
            continue

        # 제거 대상 클래스가 아직 남아있으면 스킵 (step 0-1 미실행 경고)
        if old_id not in OLD_TO_NEW:
            result["skipped_remove"] += 1
            continue

        new_id = OLD_TO_NEW[old_id]
        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))
        result["remapped"] += 1

    if not dry_run:
        with open(label_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")

    return result


def generate_new_data_yaml(dataset_root: Path, dry_run: bool = False) -> Path:
    """
    새로운 data.yaml을 생성합니다.
    기존 data.yaml이 있으면 data_original.yaml로 백업합니다.
    """
    yaml_path = dataset_root / "data.yaml"
    backup_path = dataset_root / "data_original.yaml"

    # 기존 yaml 로드
    existing_config = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            existing_config = yaml.safe_load(f) or {}

    new_config = {
        "train": existing_config.get("train", "../train/images"),
        "val": existing_config.get("val", "../valid/images"),
        "test": existing_config.get("test", "../test/images"),
        "nc": len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }

    if not dry_run:
        # 백업
        if yaml_path.exists() and not backup_path.exists():
            yaml_path.rename(backup_path)
            print(f"  📦 기존 data.yaml → data_original.yaml 백업 완료")

        with open(yaml_path, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return yaml_path


def verify_labels(dataset_root: Path) -> dict:
    """리매핑 후 라벨 검증: 모든 클래스 ID가 0~32 범위 내인지 확인합니다."""
    issues = []
    class_counter = Counter()
    total_boxes = 0

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        for label_path in labels_dir.glob("*.txt"):
            with open(label_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        issues.append(f"{label_path.name}:{line_num} 파싱 오류")
                        continue

                    if cid < 0 or cid >= len(NEW_CLASSES):
                        issues.append(
                            f"{label_path.name}:{line_num} 범위 초과 ID={cid} (유효: 0~{len(NEW_CLASSES)-1})"
                        )
                    else:
                        class_counter[NEW_CLASSES[cid]] += 1
                        total_boxes += 1

                    # bbox 좌표 범위 검증
                    if len(parts) >= 5:
                        coords = [float(p) for p in parts[1:5]]
                        for val in coords:
                            if val < 0.0 or val > 1.0:
                                issues.append(
                                    f"{label_path.name}:{line_num} bbox 좌표 범위 초과: {coords}"
                                )
                                break

    return {"issues": issues, "class_counter": class_counter, "total_boxes": total_boxes}


def main():
    parser = argparse.ArgumentParser(
        description="Step 0-2: data.yaml 수정 + 클래스 인덱스 리매핑"
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true", help="실제 변경 없이 시뮬레이션만 수행")
    args = parser.parse_args()

    root = Path(args.dataset_root)

    print("=" * 60)
    print("Step 0-2: 클래스 인덱스 리매핑 + data.yaml 수정")
    print(f"  데이터셋 경로: {root}")
    print(f"  클래스 변경: {len(ORIGINAL_CLASSES)}개 → {len(NEW_CLASSES)}개")
    print(f"  모드: {'DRY RUN' if args.dry_run else '실제 실행'}")
    print("=" * 60)

    # 매핑 테이블 출력
    print("\n── 리매핑 테이블 ──")
    for old_id, cls_name in enumerate(ORIGINAL_CLASSES):
        if cls_name in REMOVE_CLASSES:
            print(f"  {old_id:2d} ({cls_name}) → 삭제됨")
        else:
            print(f"  {old_id:2d} ({cls_name}) → {OLD_TO_NEW[old_id]:2d}")

    # data.yaml 생성
    print("\n── data.yaml 수정 ──")
    yaml_path = generate_new_data_yaml(root, dry_run=args.dry_run)
    print(f"  nc: {len(NEW_CLASSES)}")
    print(f"  names: {NEW_CLASSES[:5]}... (총 {len(NEW_CLASSES)}개)")

    # 라벨 파일 리매핑
    total = {"remapped": 0, "skipped_remove": 0, "files_processed": 0, "errors": []}

    for split in ["train", "valid", "test"]:
        labels_dir = root / split / "labels"
        if not labels_dir.exists():
            print(f"\n⚠️  {split}/labels 없음 → 건너뜀")
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"\n── {split} ({len(label_files)} 라벨 파일) ──")

        for label_path in label_files:
            result = remap_label_file(label_path, dry_run=args.dry_run)
            total["remapped"] += result["remapped"]
            total["skipped_remove"] += result["skipped_remove"]
            total["files_processed"] += 1
            total["errors"].extend(result["errors"])

        print(f"  처리 완료: {len(label_files)}개")

    print("\n" + "=" * 60)
    print("리매핑 합계:")
    print(f"  처리된 라벨 파일: {total['files_processed']}")
    print(f"  리매핑된 bbox 행: {total['remapped']}")
    if total["skipped_remove"] > 0:
        print(f"  ⚠️  제거 대상 잔존 행(step 0-1 미실행?): {total['skipped_remove']}")
    if total["errors"]:
        print(f"  ❌ 오류: {len(total['errors'])}건")
        for e in total["errors"][:10]:
            print(f"     {e}")

    # 검증
    if not args.dry_run:
        print("\n── 리매핑 후 검증 ──")
        verify = verify_labels(root)

        if verify["issues"]:
            print(f"  ❌ 문제 발견: {len(verify['issues'])}건")
            for issue in verify["issues"][:10]:
                print(f"     {issue}")
            if len(verify["issues"]) > 10:
                print(f"     ... 외 {len(verify['issues']) - 10}건")
        else:
            print(f"  ✅ 모든 클래스 ID가 0~{len(NEW_CLASSES)-1} 범위 내 정상")

        print(f"  총 bbox 수: {verify['total_boxes']}")
        print(f"\n  클래스별 bbox 수:")
        for cls_name, cnt in verify["class_counter"].most_common():
            print(f"    {cls_name}: {cnt}")

    print("=" * 60)
    if args.dry_run:
        print("\n✅ DRY RUN 완료")
    else:
        print("\n✅ Step 0-2 완료 — 다음 단계: step_0_3_oversample.py")


if __name__ == "__main__":
    main()
