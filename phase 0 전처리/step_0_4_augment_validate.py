"""
Step 0-4. 학습 전 성능 개선 작업
──────────────────────────────────
4가지 작업을 서브커맨드로 분리:
  (1) augment   : Albumentations 기반 오프라인 증강 (희소 클래스 대상)
  (2) validate  : 라벨 품질 검증 (bbox 범위, 빈 파일, 클래스 ID 유효성)
  (3) check_bg  : 배경(Negative) 이미지 현황 확인 + 생성 가이드
  (4) spotcheck : 클래스별 랜덤 샘플 시각화 (bbox 오버레이 이미지 저장)

⚠️ 반드시 step_0_3_oversample.py 실행 후에 사용하세요.
⚠️ augment 서브커맨드는 albumentations, opencv-python 패키지 필요
   → pip install albumentations opencv-python

사용법:
  python step_0_4_augment_validate.py augment   --dataset_root /path/to/dataset
  python step_0_4_augment_validate.py validate  --dataset_root /path/to/dataset
  python step_0_4_augment_validate.py check_bg  --dataset_root /path/to/dataset
  python step_0_4_augment_validate.py spotcheck --dataset_root /path/to/dataset --output_dir ./spotcheck
"""

import os
import sys
import random
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

# ── 33개 클래스 (Step 0-2 리매핑 완료 기준) ──
NEW_CLASSES = [
    "Ab_Wheel", "Aerobic_Stepper", "Arm_Curl", "Assisted_Chin_Up_Dip",
    "Back_Extension", "Barbell", "Cable_Machine", "Chest_Fly", "Chest_Press",
    "Clubbell", "Dumbbell", "Dumbbell_Rack", "Elliptical", "Foam_Roller",
    "Gym_Ball", "Hip_Abductor", "Kettlebell", "Lat_Pulldown", "Lateral_Raise",
    "Leg_Curl", "Leg_Extension", "Leg_Press", "Medicine_Ball", "Plyo_Box",
    "Punching_Bag", "Seated_Cable_Row", "Seated_Dip", "Shoulder_Press",
    "Smith_Machine", "Stationary_Bike", "T_Bar_Row", "Treadmill", "Yoga_Mat",
]

# 오버샘플링 대상 (Step 0-3 기준 300장 미만이었던 클래스)
RARE_CLASSES = {
    "Foam_Roller", "Dumbbell_Rack", "Back_Extension", "Stationary_Bike",
    "Aerobic_Stepper", "T_Bar_Row", "Ab_Wheel",
}


def find_image_file(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


def parse_yolo_labels(label_path: Path) -> list[dict]:
    """YOLO 라벨 파일을 파싱하여 [{class_id, cx, cy, w, h}, ...] 리스트를 반환합니다."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    boxes.append({
                        "class_id": int(parts[0]),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    })
                except ValueError:
                    continue
    return boxes


# ═══════════════════════════════════════════════
# (1) Albumentations 오프라인 증강
# ═══════════════════════════════════════════════
def cmd_augment(args):
    """희소 클래스 이미지에 Albumentations 증강을 적용합니다."""
    try:
        import cv2
        import albumentations as A
        import numpy as np
    except ImportError:
        print("❌ 필요 패키지 미설치:")
        print("   pip install albumentations opencv-python")
        sys.exit(1)

    root = Path(args.dataset_root)
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    n_augments = args.n_augments

    print("=" * 60)
    print("Step 0-4 (1): Albumentations 오프라인 증강")
    print(f"  대상 클래스: {sorted(RARE_CLASSES)}")
    print(f"  클래스당 증강 수: {n_augments}배")
    print(f"  모드: {'DRY RUN' if args.dry_run else '실제 실행'}")
    print("=" * 60)

    # Albumentations 파이프라인 정의
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7,
                           border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_ids"],
        min_visibility=0.3,
    ))

    # 희소 클래스 이미지 수집
    rare_stems = set()
    for label_path in train_labels.glob("*.txt"):
        boxes = parse_yolo_labels(label_path)
        for box in boxes:
            if 0 <= box["class_id"] < len(NEW_CLASSES):
                if NEW_CLASSES[box["class_id"]] in RARE_CLASSES:
                    rare_stems.add(label_path.stem)
                    break

    print(f"\n  희소 클래스 이미지 수: {len(rare_stems)}장")

    total_created = 0
    errors = 0

    for stem in sorted(rare_stems):
        img_path = find_image_file(train_images, stem)
        label_path = train_labels / f"{stem}.txt"

        if img_path is None:
            continue

        if args.dry_run:
            total_created += n_augments
            continue

        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue

        # 라벨 로드
        boxes = parse_yolo_labels(label_path)
        bboxes = [[b["cx"], b["cy"], b["w"], b["h"]] for b in boxes]
        class_ids = [b["class_id"] for b in boxes]

        for aug_i in range(n_augments):
            new_stem = f"{stem}_alb{aug_i + 1}"
            new_img_path = train_images / (new_stem + img_path.suffix)
            new_label_path = train_labels / (new_stem + ".txt")

            if new_img_path.exists():
                continue

            try:
                result = transform(image=img, bboxes=bboxes, class_ids=class_ids)
            except Exception as e:
                errors += 1
                continue

            # 증강된 이미지 저장
            cv2.imwrite(str(new_img_path), result["image"])

            # 증강된 라벨 저장
            with open(new_label_path, "w") as f:
                for bbox, cid in zip(result["bboxes"], result["class_ids"]):
                    cx, cy, w, h = bbox
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            total_created += 1

    print(f"\n  생성된 증강 이미지+라벨: {total_created}쌍")
    if errors:
        print(f"  ⚠️  오류: {errors}건")

    if args.dry_run:
        print("\n✅ DRY RUN 완료")
    else:
        print("\n✅ 증강 완료")


# ═══════════════════════════════════════════════
# (2) 라벨 품질 검증
# ═══════════════════════════════════════════════
def cmd_validate(args):
    """라벨 파일의 품질을 전수 검증합니다."""
    root = Path(args.dataset_root)

    print("=" * 60)
    print("Step 0-4 (2): 라벨 품질 검증")
    print("=" * 60)

    issues = {
        "bbox_out_of_range": [],
        "invalid_class_id": [],
        "empty_labels": [],
        "missing_labels": [],
        "missing_images": [],
        "parse_errors": [],
    }
    class_counter = Counter()
    total_boxes = 0
    hash_to_files = defaultdict(list)

    for split in ["train", "valid", "test"]:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"

        if not labels_dir.exists():
            print(f"\n⚠️  {split}/labels 없음 → 건너뜀")
            continue

        print(f"\n── {split} ──")
        image_files = set()
        label_files = set()

        # 이미지 파일 수집 + 해시 계산 (데이터 누수 검사용)
        if images_dir.exists():
            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    image_files.add(img_path.stem)
                    # 파일 해시 (처음 8KB만)
                    with open(img_path, "rb") as f:
                        h = hashlib.md5(f.read(8192)).hexdigest()
                    hash_to_files[h].append(f"{split}/{img_path.name}")

        # 라벨 파일 검증
        for label_path in sorted(labels_dir.glob("*.txt")):
            label_files.add(label_path.stem)

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                issues["empty_labels"].append(f"{split}/{label_path.name}")
                continue

            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) < 5:
                    issues["parse_errors"].append(f"{split}/{label_path.name}:{line_num} 컬럼 부족")
                    continue

                try:
                    cid = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    issues["parse_errors"].append(f"{split}/{label_path.name}:{line_num} 파싱 오류")
                    continue

                # 클래스 ID 범위
                if cid < 0 or cid >= len(NEW_CLASSES):
                    issues["invalid_class_id"].append(
                        f"{split}/{label_path.name}:{line_num} ID={cid}"
                    )
                else:
                    class_counter[NEW_CLASSES[cid]] += 1
                    total_boxes += 1

                # bbox 좌표 범위 (0~1, 약간의 여유)
                for val_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                    if val < -0.01 or val > 1.01:
                        issues["bbox_out_of_range"].append(
                            f"{split}/{label_path.name}:{line_num} {val_name}={val:.4f}"
                        )
                        break

        # 이미지-라벨 매칭
        labels_without_images = label_files - image_files
        images_without_labels = image_files - label_files

        if labels_without_images:
            for stem in list(labels_without_images)[:5]:
                issues["missing_images"].append(f"{split}/{stem}")

        print(f"  이미지: {len(image_files)}장, 라벨: {len(label_files)}개")
        print(f"  라벨만 있는 파일: {len(labels_without_images)}, 이미지만 있는 파일: {len(images_without_labels)}")

    # 데이터 누수 검사
    print("\n── 데이터 누수(Leakage) 검사 ──")
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    cross_split_leaks = []
    for h, files in duplicates.items():
        splits_involved = set(f.split("/")[0] for f in files)
        if len(splits_involved) > 1:
            cross_split_leaks.append(files)

    if cross_split_leaks:
        print(f"  ⚠️  크로스 스플릿 중복 발견: {len(cross_split_leaks)}건")
        for leak in cross_split_leaks[:5]:
            print(f"     {leak}")
    else:
        print(f"  ✅ train/valid/test 간 이미지 중복 없음")

    # 결과 리포트
    print("\n" + "=" * 60)
    print("검증 결과 요약:")
    print(f"  총 bbox 수: {total_boxes}")

    all_ok = True
    for issue_type, items in issues.items():
        if items:
            all_ok = False
            print(f"\n  ❌ {issue_type}: {len(items)}건")
            for item in items[:5]:
                print(f"     {item}")
            if len(items) > 5:
                print(f"     ... 외 {len(items) - 5}건")

    if all_ok:
        print("  ✅ 모든 검증 항목 통과!")

    print(f"\n  클래스별 bbox 분포 (전체):")
    for cls_name in NEW_CLASSES:
        cnt = class_counter.get(cls_name, 0)
        bar = "█" * max(1, cnt // 200)
        print(f"    {cls_name:25s} {cnt:>5d} {bar}")

    print("=" * 60)
    print("\n✅ Step 0-4 (2) 검증 완료")


# ═══════════════════════════════════════════════
# (3) 배경 이미지 현황 확인
# ═══════════════════════════════════════════════
def cmd_check_bg(args):
    """배경(Negative) 이미지 현황을 확인하고 가이드를 제공합니다."""
    root = Path(args.dataset_root)

    print("=" * 60)
    print("Step 0-4 (3): 배경(Negative) 이미지 현황 확인")
    print("=" * 60)

    for split in ["train", "valid", "test"]:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"

        if not labels_dir.exists():
            continue

        total = 0
        empty_labels = 0

        for label_path in labels_dir.glob("*.txt"):
            total += 1
            with open(label_path, "r") as f:
                content = f.read().strip()
            if not content:
                empty_labels += 1

        pct = (empty_labels / total * 100) if total > 0 else 0
        print(f"\n  {split}: 전체 {total}장 중 배경 이미지(빈 라벨) {empty_labels}장 ({pct:.1f}%)")

    print("\n── 권장 사항 ──")
    print("  배경 이미지(기구가 없는 헬스장 사진)를 train의 1~3% 추가하면")
    print("  False Positive를 줄이는 데 효과적입니다.")
    print("\n  추가 방법:")
    print("  1. 헬스장 배경 사진을 train/images/에 넣습니다.")
    print("  2. 각 이미지에 대응하는 빈 .txt 파일을 train/labels/에 생성합니다.")
    print("     예: background_001.jpg → background_001.txt (빈 파일)")
    print(f"\n  현재 train 기준 권장 배경 이미지 수: 약 300~900장")

    print("\n✅ Step 0-4 (3) 확인 완료")


# ═══════════════════════════════════════════════
# (4) 시각적 스팟체크
# ═══════════════════════════════════════════════
def cmd_spotcheck(args):
    """클래스별 랜덤 샘플에 bbox를 오버레이하여 이미지로 저장합니다."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ 필요 패키지 미설치:")
        print("   pip install opencv-python")
        sys.exit(1)

    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    samples_per_class = args.samples
    split = "train"

    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    print("=" * 60)
    print("Step 0-4 (4): 시각적 스팟체크")
    print(f"  클래스당 샘플 수: {samples_per_class}")
    print(f"  출력 디렉토리: {output_dir}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 클래스별 이미지 수집
    class_to_stems = defaultdict(list)
    for label_path in labels_dir.glob("*.txt"):
        boxes = parse_yolo_labels(label_path)
        for box in boxes:
            cid = box["class_id"]
            if 0 <= cid < len(NEW_CLASSES):
                class_to_stems[cid].append(label_path.stem)
                break  # 이미지당 한 번만

    # 색상 팔레트 (클래스별 고정 색상)
    random.seed(42)
    colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
              for _ in range(len(NEW_CLASSES))]

    total_saved = 0
    for cid, cls_name in enumerate(NEW_CLASSES):
        stems = class_to_stems.get(cid, [])
        if not stems:
            print(f"  ⚠️  {cls_name}: 이미지 없음")
            continue

        sampled = random.sample(stems, min(samples_per_class, len(stems)))
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(exist_ok=True)

        for stem in sampled:
            img_path = find_image_file(images_dir, stem)
            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            boxes = parse_yolo_labels(labels_dir / f"{stem}.txt")

            for box in boxes:
                bcid = box["class_id"]
                cx, cy, bw, bh = box["cx"], box["cy"], box["w"], box["h"]

                # YOLO → 픽셀 좌표
                x1 = int((cx - bw / 2) * w_img)
                y1 = int((cy - bh / 2) * h_img)
                x2 = int((cx + bw / 2) * w_img)
                y2 = int((cy + bh / 2) * h_img)

                color = colors[bcid] if 0 <= bcid < len(colors) else (0, 255, 0)
                label = NEW_CLASSES[bcid] if 0 <= bcid < len(NEW_CLASSES) else f"id={bcid}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 라벨 배경
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out_path = cls_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_path), img)
            total_saved += 1

    print(f"\n  저장된 스팟체크 이미지: {total_saved}장")
    print(f"  출력 위치: {output_dir}/")
    print(f"  각 클래스 폴더에서 bbox가 올바르게 그려졌는지 육안으로 확인하세요.")
    print("\n✅ Step 0-4 (4) 스팟체크 완료")


# ═══════════════════════════════════════════════
# 메인 진입점
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Step 0-4: 학습 전 성능 개선 작업 (증강/검증/배경확인/시각화)"
    )
    subparsers = parser.add_subparsers(dest="command", help="서브커맨드 선택")

    # (1) augment
    p_aug = subparsers.add_parser("augment", help="Albumentations 기반 오프라인 증강")
    p_aug.add_argument("--dataset_root", type=str, required=True)
    p_aug.add_argument("--n_augments", type=int, default=3, help="이미지당 증강 수 (기본: 3)")
    p_aug.add_argument("--dry_run", action="store_true")

    # (2) validate
    p_val = subparsers.add_parser("validate", help="라벨 품질 전수 검증")
    p_val.add_argument("--dataset_root", type=str, required=True)

    # (3) check_bg
    p_bg = subparsers.add_parser("check_bg", help="배경(Negative) 이미지 현황 확인")
    p_bg.add_argument("--dataset_root", type=str, required=True)

    # (4) spotcheck
    p_spot = subparsers.add_parser("spotcheck", help="클래스별 시각적 스팟체크")
    p_spot.add_argument("--dataset_root", type=str, required=True)
    p_spot.add_argument("--output_dir", type=str, default="./spotcheck", help="출력 디렉토리")
    p_spot.add_argument("--samples", type=int, default=5, help="클래스당 샘플 수 (기본: 5)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n사용 예시:")
        print("  python step_0_4_augment_validate.py augment   --dataset_root ./dataset")
        print("  python step_0_4_augment_validate.py validate  --dataset_root ./dataset")
        print("  python step_0_4_augment_validate.py check_bg  --dataset_root ./dataset")
        print("  python step_0_4_augment_validate.py spotcheck --dataset_root ./dataset")
        sys.exit(0)

    if args.command == "augment":
        cmd_augment(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "check_bg":
        cmd_check_bg(args)
    elif args.command == "spotcheck":
        cmd_spotcheck(args)


if __name__ == "__main__":
    main()
