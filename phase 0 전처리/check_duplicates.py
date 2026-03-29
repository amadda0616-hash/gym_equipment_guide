"""
check_duplicates.py — 크로스 스플릿 중복 이미지 시각 확인 스크립트

위치: gym/data/dataset/ (data.yaml과 같은 폴더)

사용법:
    python check_duplicates.py              # 중복 검사 + 샘플 10쌍 시각화 저장
    python check_duplicates.py --all        # 전체 중복 쌍 시각화
    python check_duplicates.py --delete     # 실제 중복인 test/valid 이미지 삭제 (train 유지)
"""

import argparse
import shutil
from pathlib import Path
from hashlib import md5
from collections import defaultdict
from datetime import datetime
from PIL import Image

DATASET_DIR = Path(__file__).parent
SPLITS = ["train", "valid", "test"]
OUTPUT_DIR = DATASET_DIR / "_duplicate_check"


def compute_hashes(split):
    """이미지 파일 해시 계산"""
    img_dir = DATASET_DIR / split / "images"
    if not img_dir.exists():
        return {}
    
    hashes = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in exts:
            with open(f, "rb") as fp:
                h = md5(fp.read()).hexdigest()
            hashes[f.name] = h
    return hashes


def find_cross_duplicates():
    """크로스 스플릿 중복 찾기"""
    print("해시 계산 중...")
    all_hashes = {}
    for split in SPLITS:
        hashes = compute_hashes(split)
        all_hashes[split] = hashes
        print(f"  {split}: {len(hashes)}장")

    # 역매핑: hash → [(split, filename), ...]
    hash_to_files = defaultdict(list)
    for split, hashes in all_hashes.items():
        for name, h in hashes.items():
            hash_to_files[h].append((split, name))

    # 크로스 스플릿 중복만 추출
    duplicates = []
    for h, files in hash_to_files.items():
        splits = set(s for s, _ in files)
        if len(splits) > 1:
            duplicates.append(files)

    print(f"\n크로스 스플릿 중복: {len(duplicates)}건")
    return duplicates


def save_comparison_images(duplicates, max_pairs=None):
    """중복 쌍을 나란히 비교 이미지로 저장"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 기존 파일 정리
    for f in OUTPUT_DIR.glob("*.jpg"):
        f.unlink()

    pairs = duplicates if max_pairs is None else duplicates[:max_pairs]
    saved = 0

    for idx, files in enumerate(pairs):
        try:
            images = []
            labels = []
            for split, name in files:
                img_path = DATASET_DIR / split / "images" / name
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    labels.append(f"{split}/{name[:40]}")

            if len(images) < 2:
                continue

            # 나란히 합치기
            widths = [img.width for img in images]
            max_h = max(img.height for img in images)
            total_w = sum(widths) + 10 * (len(images) - 1)  # 간격 10px

            canvas = Image.new("RGB", (total_w, max_h + 30), (0, 0, 0))
            x_offset = 0
            for img, label in zip(images, labels):
                canvas.paste(img, (x_offset, 30))
                x_offset += img.width + 10

            # 레이블은 파일명으로 대체 (PIL에 폰트 없이)
            out_name = f"dup_{idx:03d}_{'_vs_'.join(s for s, _ in files)}.jpg"
            canvas.save(OUTPUT_DIR / out_name, quality=85)
            saved += 1

            # 터미널에도 출력
            file_strs = [f"{s}/{n[:50]}" for s, n in files]
            same_size = len(set(img.size for img in images)) == 1
            print(f"  [{idx:3d}] {'✅ 동일크기' if same_size else '⚠️ 크기다름'} {' ↔ '.join(file_strs)}")

        except Exception as e:
            print(f"  [{idx:3d}] 오류: {e}")

    print(f"\n📸 비교 이미지 {saved}장 → {OUTPUT_DIR.resolve()}")
    return saved


def delete_duplicates(duplicates):
    """중복 이미지 삭제 (train 유지, valid/test에서 제거)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = DATASET_DIR / f"_dup_removed_{timestamp}"
    backup_dir.mkdir(exist_ok=True)

    removed = 0
    for files in duplicates:
        splits = set(s for s, _ in files)
        if len(splits) <= 1:
            continue

        # train에 있는 이미지는 유지, valid/test의 중복만 제거
        for split, name in files:
            if split in ("valid", "test"):
                img_path = DATASET_DIR / split / "images" / name
                lbl_path = DATASET_DIR / split / "labels" / (Path(name).stem + ".txt")

                if img_path.exists():
                    shutil.move(str(img_path), str(backup_dir / f"{split}__{name}"))
                    if lbl_path.exists():
                        shutil.move(str(lbl_path), str(backup_dir / f"{split}__{lbl_path.name}"))
                    removed += 1

    # .cache 삭제
    for cache in DATASET_DIR.rglob("*.cache"):
        cache.unlink()
        print(f"  🗑️ 캐시 삭제: {cache.name}")

    print(f"\n✅ {removed}장 삭제 (valid/test에서)")
    print(f"📦 백업: {backup_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="크로스 스플릿 중복 이미지 확인")
    parser.add_argument("--all", action="store_true", help="전체 중복 쌍 시각화 (기본: 10쌍)")
    parser.add_argument("--delete", action="store_true", help="valid/test의 중복 이미지 삭제")
    args = parser.parse_args()

    print("=" * 60)
    print("🔍 크로스 스플릿 중복 이미지 확인")
    print(f"   데이터셋: {DATASET_DIR.resolve()}")
    print("=" * 60)

    duplicates = find_cross_duplicates()

    if not duplicates:
        print("✅ 크로스 스플릿 중복 없음!")
        return

    # 시각화
    max_pairs = None if args.all else 10
    save_comparison_images(duplicates, max_pairs)

    if args.delete:
        print(f"\n🔴 valid/test의 중복 이미지를 삭제합니다...")
        delete_duplicates(duplicates)
    else:
        print(f"\n💡 다음 단계:")
        print(f"   1. {OUTPUT_DIR.resolve()} 폴더에서 비교 이미지 확인")
        print(f"   2. 실제 중복이면: python check_duplicates.py --delete")
        print(f"   3. 오탐이면: 무시하고 학습 진행")


if __name__ == "__main__":
    main()
