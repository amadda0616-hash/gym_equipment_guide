"""
Step 0-3. 희소 클래스 오버샘플링
──────────────────────────────────
- 300장 미만인 클래스에 대해 이미지+라벨 파일을 복사하여 증강
- 복사본에 접미사(_os1, _os2, ...)를 붙여 원본과 구분
- train split에만 적용 (valid/test는 원본 유지)

⚠️ 반드시 step_0_2_remap_classes.py 실행 후에 사용하세요.

사용법:
  python step_0_3_oversample.py --dataset_root /path/to/dataset
  python step_0_3_oversample.py --dataset_root /path/to/dataset --target 300
"""

import os
import math
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict


# ── 리매핑 완료된 33개 클래스 ──
NEW_CLASSES = [
    "Ab_Wheel", "Aerobic_Stepper", "Arm_Curl", "Assisted_Chin_Up_Dip",
    "Back_Extension", "Barbell", "Cable_Machine", "Chest_Fly", "Chest_Press",
    "Clubbell", "Dumbbell", "Dumbbell_Rack", "Elliptical", "Foam_Roller",
    "Gym_Ball", "Hip_Abductor", "Kettlebell", "Lat_Pulldown", "Lateral_Raise",
    "Leg_Curl", "Leg_Extension", "Leg_Press", "Medicine_Ball", "Plyo_Box",
    "Punching_Bag", "Seated_Cable_Row", "Seated_Dip", "Shoulder_Press",
    "Smith_Machine", "Stationary_Bike", "T_Bar_Row", "Treadmill", "Yoga_Mat",
]


def scan_class_distribution(labels_dir: Path) -> tuple[Counter, dict]:
    """
    라벨 디렉토리를 스캔하여 클래스별 이미지 수와 이미지-클래스 매핑을 반환합니다.

    Returns:
        class_image_count: Counter {class_name: 이미지_등장_수}
        class_to_images: dict {class_id: [label_stem1, label_stem2, ...]}
    """
    class_image_count = Counter()
    class_to_images = defaultdict(list)

    for label_path in sorted(labels_dir.glob("*.txt")):
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            continue

        # 이 이미지에 포함된 클래스 ID (중복 제거)
        class_ids = set()
        for line in lines:
            parts = line.split()
            if parts:
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    continue

        stem = label_path.stem
        for cid in class_ids:
            if 0 <= cid < len(NEW_CLASSES):
                class_image_count[NEW_CLASSES[cid]] += 1
                class_to_images[cid].append(stem)

    return class_image_count, class_to_images


def find_image_file(images_dir: Path, stem: str) -> Path | None:
    """라벨 stem에 대응하는 이미지 파일을 찾습니다."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        img_path = images_dir / (stem + ext)
        if img_path.exists():
            return img_path
    return None


def oversample_class(
    class_id: int,
    class_name: str,
    image_stems: list[str],
    images_dir: Path,
    labels_dir: Path,
    target: int,
    dry_run: bool = False,
) -> dict:
    """
    특정 클래스의 이미지+라벨을 복사하여 오버샘플링합니다.

    target 장에 도달할 때까지 이미지를 순환하며 복사합니다.
    """
    current = len(image_stems)
    if current >= target:
        return {"class": class_name, "before": current, "added": 0, "after": current}

    copies_needed = target - current
    added = 0

    for i in range(copies_needed):
        # 원본 이미지를 순환하며 선택
        src_stem = image_stems[i % current]
        suffix = f"_os{i + 1}"
        new_stem = src_stem + suffix

        # 원본 이미지 파일 찾기
        src_img = find_image_file(images_dir, src_stem)
        src_label = labels_dir / f"{src_stem}.txt"

        if src_img is None or not src_label.exists():
            continue

        # 복사 대상 경로
        new_img = images_dir / (new_stem + src_img.suffix)
        new_label = labels_dir / (new_stem + ".txt")

        # 이미 존재하면 건너뜀
        if new_img.exists() or new_label.exists():
            continue

        if not dry_run:
            shutil.copy2(src_img, new_img)
            shutil.copy2(src_label, new_label)

        added += 1

    return {
        "class": class_name,
        "before": current,
        "added": added,
        "after": current + added,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 0-3: 희소 클래스 오버샘플링 (이미지+라벨 복사)"
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--target", type=int, default=300,
        help="오버샘플링 목표 이미지 수 (기본: 300)"
    )
    parser.add_argument("--dry_run", action="store_true", help="실제 변경 없이 시뮬레이션만 수행")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"

    if not train_labels.exists():
        print(f"❌ train/labels 디렉토리를 찾을 수 없습니다: {train_labels}")
        return

    print("=" * 60)
    print("Step 0-3: 희소 클래스 오버샘플링")
    print(f"  데이터셋 경로: {root}")
    print(f"  목표 이미지 수: {args.target}장")
    print(f"  모드: {'DRY RUN' if args.dry_run else '실제 실행'}")
    print("=" * 60)

    # 현재 분포 스캔
    print("\n── 현재 train 클래스 분포 스캔 ──")
    class_counts, class_to_images = scan_class_distribution(train_labels)

    # 오버샘플링 대상 식별
    undersample_classes = []
    for cid, cls_name in enumerate(NEW_CLASSES):
        count = class_counts.get(cls_name, 0)
        if count < args.target:
            undersample_classes.append((cid, cls_name, count))

    if not undersample_classes:
        print(f"\n✅ 모든 클래스가 {args.target}장 이상입니다. 오버샘플링 불필요.")
        return

    print(f"\n  오버샘플링 대상: {len(undersample_classes)}개 클래스")
    for cid, name, cnt in undersample_classes:
        multiplier = math.ceil(args.target / cnt) if cnt > 0 else "∞"
        print(f"    {name}: {cnt}장 (×{multiplier} 필요)")

    # 오버샘플링 실행
    print(f"\n── 오버샘플링 {'시뮬레이션' if args.dry_run else '실행'} ──")
    results = []

    for cid, cls_name, current_count in undersample_classes:
        stems = class_to_images.get(cid, [])
        if not stems:
            print(f"  ⚠️  {cls_name}: 이미지 stem을 찾을 수 없음 → 건너뜀")
            continue

        result = oversample_class(
            class_id=cid,
            class_name=cls_name,
            image_stems=stems,
            images_dir=train_images,
            labels_dir=train_labels,
            target=args.target,
            dry_run=args.dry_run,
        )
        results.append(result)
        print(f"  {result['class']}: {result['before']}장 → +{result['added']} → {result['after']}장")

    # 결과 요약
    print("\n" + "=" * 60)
    print("오버샘플링 결과 요약:")
    total_added = sum(r["added"] for r in results)
    print(f"  처리된 클래스: {len(results)}개")
    print(f"  추가된 이미지+라벨 쌍: {total_added}개")

    # 최종 분포 확인 (dry_run이 아닌 경우)
    if not args.dry_run and total_added > 0:
        print("\n── 오버샘플링 후 최종 분포 ──")
        final_counts, _ = scan_class_distribution(train_labels)
        for cls_name in NEW_CLASSES:
            cnt = final_counts.get(cls_name, 0)
            marker = " ⬆️" if any(r["class"] == cls_name and r["added"] > 0 for r in results) else ""
            print(f"  {cls_name}: {cnt}장{marker}")

        new_max = max(final_counts.values())
        new_min = min(final_counts.values()) if final_counts else 0
        if new_min > 0:
            print(f"\n  최종 불균형 비율: {new_max}:{new_min} = {new_max / new_min:.0f}:1")

    print("=" * 60)
    if args.dry_run:
        print("\n✅ DRY RUN 완료")
    else:
        print("\n✅ Step 0-3 완료 — 다음 단계: step_0_4_augment_validate.py")


if __name__ == "__main__":
    main()
