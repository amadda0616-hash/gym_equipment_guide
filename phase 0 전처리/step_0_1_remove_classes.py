"""
Step 0-1. 극소 클래스 4개 제거
─────────────────────────────────
- Resistance_Band (2장), Jump_Rope (6장), Weight_Plates (12장), Person (22장)
- 해당 클래스의 bbox 행만 라벨 파일에서 삭제
- 제거 후 라벨이 완전히 비게 되는 이미지는 삭제(이미지+라벨 모두)

사용법:
  python step_0_1_remove_classes.py --dataset_root /path/to/dataset

dataset_root 구조 (Roboflow 표준):
  dataset_root/
    ├── train/  (또는 data.yaml의 train 경로)
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

import os
import glob
import argparse
from pathlib import Path


# ── 현재 data.yaml 기준 37개 클래스 ──
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

# 제거 대상 클래스
REMOVE_CLASSES = {"Jump_Rope", "Person", "Resistance_Band", "Weight_Plates"}

# 제거 대상 클래스 인덱스
REMOVE_IDS = {i for i, name in enumerate(ORIGINAL_CLASSES) if name in REMOVE_CLASSES}


def process_label_file(label_path: Path) -> dict:
    """
    라벨 파일에서 제거 대상 클래스의 bbox 행을 삭제합니다.

    Returns:
        dict with keys:
          - 'removed_lines': 삭제된 행 수
          - 'remaining_lines': 남은 행 수
          - 'empty_after': 제거 후 빈 파일이 되었는지
    """
    if not label_path.exists():
        return {"removed_lines": 0, "remaining_lines": 0, "empty_after": True}

    with open(label_path, "r") as f:
        lines = f.readlines()

    # 빈 파일 처리
    original_count = len([l for l in lines if l.strip()])
    if original_count == 0:
        return {"removed_lines": 0, "remaining_lines": 0, "empty_after": True}

    kept_lines = []
    removed_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        class_id = int(stripped.split()[0])
        if class_id in REMOVE_IDS:
            removed_count += 1
        else:
            kept_lines.append(stripped)

    remaining = len(kept_lines)

    # 라벨 파일 덮어쓰기
    with open(label_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    return {
        "removed_lines": removed_count,
        "remaining_lines": remaining,
        "empty_after": remaining == 0,
    }


def find_image_file(images_dir: Path, stem: str) -> Path | None:
    """라벨 파일명(stem)에 대응하는 이미지 파일을 찾습니다."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        img_path = images_dir / (stem + ext)
        if img_path.exists():
            return img_path
    return None


def process_split(split_dir: Path, dry_run: bool = False) -> dict:
    """
    하나의 split(train/valid/test) 디렉토리를 처리합니다.
    """
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"

    if not labels_dir.exists():
        print(f"  ⚠️  라벨 디렉토리 없음: {labels_dir}")
        return {}

    label_files = sorted(labels_dir.glob("*.txt"))

    stats = {
        "total_labels": len(label_files),
        "modified": 0,
        "deleted_empty": 0,
        "deleted_images": 0,
        "total_removed_lines": 0,
    }

    for label_path in label_files:
        result = process_label_file(label_path) if not dry_run else _dry_run_check(label_path)

        if result["removed_lines"] > 0:
            stats["modified"] += 1
            stats["total_removed_lines"] += result["removed_lines"]

        # 제거 후 빈 파일 → 이미지와 라벨 모두 삭제
        if result["empty_after"] and result["removed_lines"] > 0:
            stats["deleted_empty"] += 1
            img_path = find_image_file(images_dir, label_path.stem)

            if not dry_run:
                label_path.unlink()
                if img_path and img_path.exists():
                    img_path.unlink()
                    stats["deleted_images"] += 1
            else:
                if img_path:
                    stats["deleted_images"] += 1

    return stats


def _dry_run_check(label_path: Path) -> dict:
    """dry_run 모드: 파일을 변경하지 않고 결과만 계산합니다."""
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    removed = sum(1 for l in lines if int(l.split()[0]) in REMOVE_IDS)
    remaining = len(lines) - removed

    return {
        "removed_lines": removed,
        "remaining_lines": remaining,
        "empty_after": remaining == 0 and removed > 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 0-1: 극소 클래스 4개(Resistance_Band, Jump_Rope, Weight_Plates, Person) 라벨 제거"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="데이터셋 루트 디렉토리 (train/, valid/, test/ 폴더가 있는 경로)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="실제 변경 없이 시뮬레이션만 수행"
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    splits = ["train", "valid", "test"]

    print("=" * 60)
    print("Step 0-1: 극소 클래스 제거")
    print(f"  제거 대상: {sorted(REMOVE_CLASSES)}")
    print(f"  제거 대상 ID: {sorted(REMOVE_IDS)}")
    print(f"  데이터셋 경로: {root}")
    print(f"  모드: {'DRY RUN (변경 없음)' if args.dry_run else '실제 실행'}")
    print("=" * 60)

    total_stats = {"modified": 0, "deleted_empty": 0, "deleted_images": 0, "total_removed_lines": 0}

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"\n⚠️  {split}/ 디렉토리 없음 → 건너뜀")
            continue

        print(f"\n── {split} ──")
        stats = process_split(split_dir, dry_run=args.dry_run)

        if stats:
            print(f"  라벨 파일 수: {stats['total_labels']}")
            print(f"  수정된 라벨: {stats['modified']}")
            print(f"  삭제된 bbox 행: {stats['total_removed_lines']}")
            print(f"  빈 파일 삭제(라벨): {stats['deleted_empty']}")
            print(f"  빈 파일 삭제(이미지): {stats['deleted_images']}")

            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

    print("\n" + "=" * 60)
    print("합계:")
    print(f"  수정된 라벨 파일: {total_stats['modified']}")
    print(f"  삭제된 bbox 행 총계: {total_stats['total_removed_lines']}")
    print(f"  삭제된 빈 이미지+라벨 쌍: {total_stats['deleted_empty']}")
    print("=" * 60)

    if args.dry_run:
        print("\n✅ DRY RUN 완료 — 실제 파일은 변경되지 않았습니다.")
        print("   실제 실행하려면 --dry_run 플래그를 제거하세요.")
    else:
        print("\n✅ Step 0-1 완료 — 다음 단계: step_0_2_remap_classes.py")


if __name__ == "__main__":
    main()
