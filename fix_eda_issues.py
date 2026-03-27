"""
Phase 0 추가 수정 — EDA 결과 기반
──────────────────────────────────
EDA에서 발견된 문제를 해결합니다:

  (1) 세그멘테이션 라벨 혼입 정리 (Warning 해결)
      - 라벨 파일 5개에 폴리곤 좌표가 섞여 있어 bbox(5컬럼)로 자름
      
  (2) Foam_Roller / Dumbbell_Rack valid/test 재분배
      - Foam_Roller: valid 0장, test 0장 → 원본에서 각 7장 이동
      - Dumbbell_Rack: valid 9장 → 원본에서 valid로 5장 추가 이동
      
  (3) .cache 파일 삭제 (수정 반영)

⚠️ Phase 0 (step_0_1 ~ step_0_4) 완료 후, 학습 전에 실행하세요.

사용법:
  cd C:\\Users\\user\\github\\yolo26_app\\gym\\data\\dataset
  python fix_eda_issues.py --dry_run     # 시뮬레이션
  python fix_eda_issues.py               # 실제 실행
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


# ── 33개 클래스 (리매핑 완료 기준) ──
NEW_CLASSES = [
    "Ab_Wheel", "Aerobic_Stepper", "Arm_Curl", "Assisted_Chin_Up_Dip",
    "Back_Extension", "Barbell", "Cable_Machine", "Chest_Fly", "Chest_Press",
    "Clubbell", "Dumbbell", "Dumbbell_Rack", "Elliptical", "Foam_Roller",
    "Gym_Ball", "Hip_Abductor", "Kettlebell", "Lat_Pulldown", "Lateral_Raise",
    "Leg_Curl", "Leg_Extension", "Leg_Press", "Medicine_Ball", "Plyo_Box",
    "Punching_Bag", "Seated_Cable_Row", "Seated_Dip", "Shoulder_Press",
    "Smith_Machine", "Stationary_Bike", "T_Bar_Row", "Treadmill", "Yoga_Mat",
]


def find_image_file(images_dir: Path, stem: str) -> Path | None:
    """라벨 stem에 대응하는 이미지 파일을 찾습니다."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════
# (1) 세그멘테이션 라벨 정리
# ═══════════════════════════════════════════════
def fix_segment_labels(labels_dir: Path, dry_run: bool = False) -> int:
    """라벨 파일에서 세그멘테이션 좌표(5컬럼 초과)를 제거하고 bbox만 남깁니다."""
    fixed = 0
    for label_path in labels_dir.glob("*.txt"):
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:  # class_id cx cy w h 이후 추가 좌표 존재
                new_lines.append(" ".join(parts[:5]))
                modified = True
            elif parts:
                new_lines.append(line.strip())

        if modified:
            fixed += 1
            if not dry_run:
                with open(label_path, "w") as f:
                    for nl in new_lines:
                        f.write(nl + "\n")
    return fixed


# ═══════════════════════════════════════════════
# (2) valid/test 재분배
# ═══════════════════════════════════════════════
def get_class_stems(labels_dir: Path, target_class_id: int) -> list[str]:
    """특정 클래스를 포함하는 원본 이미지(오버샘플링/증강 아닌)의 stem 목록을 반환합니다."""
    stems = []
    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        # 오버샘플링/증강본 제외 — 원본만
        if "_os" in stem or "_alb" in stem:
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(float(parts[0]))
                        if cid == target_class_id:
                            stems.append(stem)
                            break
                    except ValueError:
                        continue
    return stems


def move_images_between_splits(
    src_split: str,
    dst_split: str,
    stems: list[str],
    dataset_root: Path,
    dry_run: bool = False,
) -> int:
    """이미지+라벨 쌍을 src_split에서 dst_split으로 이동합니다."""
    src_images = dataset_root / src_split / "images"
    src_labels = dataset_root / src_split / "labels"
    dst_images = dataset_root / dst_split / "images"
    dst_labels = dataset_root / dst_split / "labels"

    moved = 0
    for stem in stems:
        img = find_image_file(src_images, stem)
        lbl = src_labels / f"{stem}.txt"

        if img is None or not lbl.exists():
            continue

        dst_img = dst_images / img.name
        dst_lbl = dst_labels / lbl.name

        # 이미 존재하면 건너뜀
        if dst_img.exists() or dst_lbl.exists():
            continue

        if not dry_run:
            shutil.move(str(img), str(dst_img))
            shutil.move(str(lbl), str(dst_lbl))

            # 오버샘플링/증강본도 함께 삭제 (train에서만)
            if src_split == "train":
                for suffix_pattern in [f"{stem}_os*", f"{stem}_alb*"]:
                    for aug_img in src_images.glob(suffix_pattern + ".*"):
                        aug_img.unlink()
                    for aug_lbl in src_labels.glob(suffix_pattern + ".txt"):
                        aug_lbl.unlink()

        moved += 1
    return moved


# ═══════════════════════════════════════════════
# (3) 캐시 삭제
# ═══════════════════════════════════════════════
def delete_caches(dataset_root: Path, dry_run: bool = False) -> int:
    """YOLO .cache 파일을 삭제합니다."""
    deleted = 0
    for cache_file in dataset_root.rglob("*.cache"):
        if not dry_run:
            cache_file.unlink()
        deleted += 1
    return deleted


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="EDA 결과 기반 Phase 0 추가 수정")
    parser.add_argument("--dataset_root", type=str, default=".",
                        help="데이터셋 루트 경로 (기본: 현재 디렉토리)")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제 변경 없이 시뮬레이션만 수행")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    random.seed(42)

    print("=" * 60)
    print("EDA 결과 기반 Phase 0 추가 수정")
    print(f"  데이터셋 경로: {root.resolve()}")
    print(f"  모드: {'DRY RUN' if args.dry_run else '실제 실행'}")
    print("=" * 60)

    # ── (1) 세그멘테이션 라벨 정리 ──
    print("\n── (1) 세그멘테이션 라벨 정리 ──")
    for split in ["train", "valid", "test"]:
        labels_dir = root / split / "labels"
        if labels_dir.exists():
            count = fix_segment_labels(labels_dir, dry_run=args.dry_run)
            print(f"  {split}: {count}개 파일에서 세그멘테이션 좌표 제거")

    # ── (2) Foam_Roller valid/test 재분배 ──
    print("\n── (2) Foam_Roller valid/test 재분배 ──")
    foam_id = NEW_CLASSES.index("Foam_Roller")  # 13
    train_labels = root / "train" / "labels"

    foam_stems = get_class_stems(train_labels, foam_id)
    random.shuffle(foam_stems)
    print(f"  Foam_Roller 원본 train 이미지: {len(foam_stems)}장")

    if len(foam_stems) >= 14:
        to_valid = foam_stems[:7]
        to_test = foam_stems[7:14]

        moved_v = move_images_between_splits("train", "valid", to_valid, root, dry_run=args.dry_run)
        moved_t = move_images_between_splits("train", "test", to_test, root, dry_run=args.dry_run)
        print(f"  → valid로 이동: {moved_v}장")
        print(f"  → test로 이동: {moved_t}장")
        print(f"  → train 원본 남음: {len(foam_stems) - moved_v - moved_t}장 (+ 증강본)")
    else:
        print(f"  ⚠️ 원본이 {len(foam_stems)}장뿐이라 이동 불가 — 수동 확인 필요")

    # ── (2-b) Dumbbell_Rack valid 보충 ──
    print("\n── (2-b) Dumbbell_Rack valid 보충 ──")
    rack_id = NEW_CLASSES.index("Dumbbell_Rack")  # 11

    rack_stems = get_class_stems(train_labels, rack_id)
    random.shuffle(rack_stems)
    print(f"  Dumbbell_Rack 원본 train 이미지: {len(rack_stems)}장")

    if len(rack_stems) >= 5:
        to_valid_rack = rack_stems[:5]
        moved_rv = move_images_between_splits("train", "valid", to_valid_rack, root, dry_run=args.dry_run)
        print(f"  → valid로 이동: {moved_rv}장 (기존 9장 + {moved_rv}장 = {9 + moved_rv}장)")
    else:
        print(f"  ⚠️ 원본이 {len(rack_stems)}장뿐이라 이동 불가")

    # ── (3) 캐시 삭제 ──
    print("\n── (3) .cache 파일 삭제 ──")
    deleted = delete_caches(root, dry_run=args.dry_run)
    print(f"  삭제된 캐시 파일: {deleted}개")

    # ── 결과 요약 ──
    print("\n" + "=" * 60)
    if args.dry_run:
        print("✅ DRY RUN 완료 — 실제 파일은 변경되지 않았습니다.")
    else:
        print("✅ EDA 수정 완료")
        print("   다음 단계: 노트북에서 Phase 2 학습 셀 실행")
    print("=" * 60)


if __name__ == "__main__":
    main()
