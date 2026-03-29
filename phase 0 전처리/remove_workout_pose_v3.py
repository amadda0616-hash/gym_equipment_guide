"""
remove_workout_pose_v3.py — 운동 자세 / 노이즈 이미지 삭제 스크립트

위치: gym/data/dataset/ (data.yaml과 같은 폴더)

사용법:
    python remove_workout_pose_v3.py                    # 드라이런 (리포트만)
    python remove_workout_pose_v3.py --visual-check 30  # 삭제 대상 30장 샘플 저장
    python remove_workout_pose_v3.py --execute          # 실제 삭제 (백업 후)

삭제 기준 (3중 AND — 세 조건 모두 해당해야 삭제):
    1. 검은 패딩 (letterbox) — 세로 영상 프레임에서 추출된 이미지
    2. 모든 bbox 소형 — 면적 3% 미만 (기구가 아닌 손에 쥔 도구 수준)
    3. bbox 최대 변 길이 15% 미만 — 아주 작은 객체만 있는 경우

    → 기구가 크게 찍힌 정상 이미지는 조건 2, 3에서 걸러져 보호됩니다.
    → 검은 패딩 없는 일반 배경 이미지도 조건 1에서 걸러져 보호됩니다.
"""

import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from PIL import Image
import argparse

# ═══════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════

DATASET_DIR = Path(__file__).parent  # 스크립트와 같은 폴더 (dataset/)
SPLITS = ["train", "valid", "test"]

# 삭제 기준 파라미터
BLACK_BAR_WIDTH = 30            # 검사할 가장자리 픽셀 두께
BLACK_BAR_THRESHOLD = 15        # 가장자리 평균 밝기 (이하면 검은 패딩)
BBOX_AREA_THRESHOLD = 0.03      # bbox 면적 3% 미만이면 소형
BBOX_MAX_DIM_THRESHOLD = 0.15   # bbox의 w, h 모두 15% 미만이면 소형


# ═══════════════════════════════════════════
# 검출 함수
# ═══════════════════════════════════════════

def has_black_bars(img_path):
    """좌우에 검은 패딩(letterbox)이 있는지 검사"""
    try:
        img = Image.open(img_path).convert("L")  # 그레이스케일
        arr = np.array(img)
        h, w = arr.shape

        if w < 100 or h < 100:
            return False

        left = arr[:, :BLACK_BAR_WIDTH].mean()
        right = arr[:, -BLACK_BAR_WIDTH:].mean()

        return left < BLACK_BAR_THRESHOLD and right < BLACK_BAR_THRESHOLD
    except Exception:
        return False


def has_only_tiny_bboxes(label_path):
    """라벨의 모든 bbox가 아주 작은지 검사 (면적 AND 최대변 기준)"""
    if not label_path.exists():
        return False

    try:
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            return False  # 빈 라벨(배경 이미지)은 보호

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            bw = float(parts[3])
            bh = float(parts[4])
            area = bw * bh

            # 하나라도 큰 bbox가 있으면 정상 이미지
            if area >= BBOX_AREA_THRESHOLD:
                return False
            if bw >= BBOX_MAX_DIM_THRESHOLD or bh >= BBOX_MAX_DIM_THRESHOLD:
                return False

        return True  # 모든 bbox가 아주 작음
    except Exception:
        return False


def get_class_ids(label_path, class_names):
    """라벨에서 클래스명 목록 반환"""
    result = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(float(parts[0]))
                    name = class_names[cid] if cid < len(class_names) else f"id_{cid}"
                    result.append(name)
    return result


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="운동 자세/노이즈 이미지 삭제 v3")
    parser.add_argument("--execute", action="store_true", help="실제 삭제 (백업 후)")
    parser.add_argument("--visual-check", type=int, default=0, metavar="N",
                        help="삭제 대상 N장을 샘플 폴더에 저장")
    args = parser.parse_args()

    # data.yaml 로드
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"❌ data.yaml을 찾을 수 없습니다: {yaml_path}")
        print(f"   이 스크립트를 dataset 폴더에서 실행하세요.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])

    print("=" * 60)
    print("🏋️ Workout Pose 이미지 삭제 v3 (3중 AND)")
    print(f"   데이터셋: {DATASET_DIR.resolve()}")
    print(f"   조건 1: 검은 패딩 (좌우 {BLACK_BAR_WIDTH}px 평균 < {BLACK_BAR_THRESHOLD})")
    print(f"   조건 2: 소형 bbox (면적 < {BBOX_AREA_THRESHOLD*100:.0f}%)")
    print(f"   조건 3: bbox 최대변 < {BBOX_MAX_DIM_THRESHOLD*100:.0f}%")
    print(f"   삭제 조건: 1 AND 2 AND 3 (세 조건 모두 해당)")
    print(f"   모드: {'🔴 실제 삭제' if args.execute else '🟢 드라이런'}")
    print("=" * 60)

    # 스캔
    targets = []  # (img_path, label_path, split, class_list)
    stats = {}
    class_counter = Counter()

    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        if not img_dir.exists():
            print(f"  ⚠️ {split}/images 없음")
            continue

        imgs = sorted([f for f in img_dir.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])

        print(f"\n🔍 {split} 스캔 중... ({len(imgs)}장)")

        black_bar_cnt = 0
        tiny_bbox_cnt = 0
        both_cnt = 0

        for img_path in imgs:
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            is_padded = has_black_bars(img_path)
            is_tiny = has_only_tiny_bboxes(lbl_path)

            if is_padded:
                black_bar_cnt += 1
            if is_tiny:
                tiny_bbox_cnt += 1

            # 3중 AND: 패딩 + 소형bbox (면적 + 최대변 모두 검사는 has_only_tiny_bboxes 안에 통합)
            if is_padded and is_tiny:
                both_cnt += 1
                cls_list = get_class_ids(lbl_path, class_names)
                targets.append((img_path, lbl_path, split, cls_list))
                for c in cls_list:
                    class_counter[c] += 1

        stats[split] = {
            "total": len(imgs),
            "black_bar": black_bar_cnt,
            "tiny_bbox": tiny_bbox_cnt,
            "delete": both_cnt,
        }

        print(f"  전체: {len(imgs)} | 검은패딩: {black_bar_cnt} | 소형bbox: {tiny_bbox_cnt} | ✂️ 삭제대상(AND): {both_cnt}")

    # 리포트
    total_imgs = sum(s["total"] for s in stats.values())
    total_del = len(targets)

    print("\n" + "=" * 60)
    print("📊 삭제 리포트")
    print("=" * 60)
    print(f"전체 이미지: {total_imgs}장")
    pct = (total_del / total_imgs * 100) if total_imgs > 0 else 0
    print(f"삭제 대상:   {total_del}장 ({pct:.1f}%)")

    print(f"\n{'Split':<8} {'전체':>8} {'검은패딩':>8} {'소형bbox':>8} {'삭제':>8}")
    print("-" * 44)
    for split in SPLITS:
        s = stats.get(split)
        if s:
            print(f"{split:<8} {s['total']:>8} {s['black_bar']:>8} {s['tiny_bbox']:>8} {s['delete']:>8}")

    if class_counter:
        print(f"\n클래스별 삭제 이미지 수:")
        for cls, cnt in class_counter.most_common():
            warn = " ⚠️" if cnt > 200 else ""
            print(f"  {cls:<25} {cnt}장{warn}")

    # 시각 확인
    if args.visual_check > 0 and targets:
        import random
        check_dir = DATASET_DIR / "_visual_check_v3"
        check_dir.mkdir(exist_ok=True)
        for f in check_dir.glob("*"):
            f.unlink()

        samples = random.sample(targets, min(args.visual_check, len(targets)))
        for img_path, lbl_path, split, cls_list in samples:
            cls_str = "_".join(cls_list[:2]) if cls_list else "empty"
            shutil.copy2(img_path, check_dir / f"{split}__{cls_str}__{img_path.name}")

        print(f"\n📸 시각 확인: {len(samples)}장 → {check_dir.resolve()}")

    # 실행
    if args.execute and targets:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = DATASET_DIR / f"_removed_v3_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        (backup_dir / "images").mkdir(exist_ok=True)
        (backup_dir / "labels").mkdir(exist_ok=True)

        removed = 0
        for img_path, lbl_path, split, cls_list in targets:
            try:
                if img_path.exists():
                    shutil.move(str(img_path), str(backup_dir / "images" / f"{split}__{img_path.name}"))
                if lbl_path.exists():
                    shutil.move(str(lbl_path), str(backup_dir / "labels" / f"{split}__{lbl_path.name}"))
                removed += 1
            except Exception as e:
                print(f"  삭제 실패: {img_path.name} ({e})")

        # .cache 삭제
        for cache in DATASET_DIR.rglob("*.cache"):
            cache.unlink()
            print(f"  🗑️ 캐시 삭제: {cache.name}")

        print(f"\n✅ 삭제 완료: {removed}장")
        print(f"📦 백업: {backup_dir.resolve()}")
        print(f"   (복원: 백업 폴더에서 원래 위치로 이동)")

    elif not args.execute:
        print(f"\n🟢 드라이런 완료.")
        print(f"   시각 확인: python remove_workout_pose_v3.py --visual-check 30")
        print(f"   실제 삭제: python remove_workout_pose_v3.py --execute")


if __name__ == "__main__":
    main()
