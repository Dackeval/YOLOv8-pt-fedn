import argparse, os, shutil, glob, random, hashlib
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")
LBL_EXT = ".txt"

def find_pairs(root: Path):
    """Return list of (img_path, lbl_path, rel_parent_str)."""
    root = root.resolve()
    labels = [Path(p) for p in glob.glob(str(root / "**" / f"*{LBL_EXT}"), recursive=True)]
    pairs, missing = [], 0
    for lp in labels:
        base = lp.stem
        img = None
        for ext in IMG_EXTS:
            cand = lp.with_name(base + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            missing += 1
            continue
        rel_parent = str(lp.parent.relative_to(root))  # e.g. "200614_SuddenValley_Phantom_VIS_0006"
        pairs.append((img.resolve(), lp.resolve(), rel_parent))
    if missing:
        print(f"[warn] {missing} labels had no matching image and were skipped")
    return pairs

def safe_name(img: Path, rel_parent: str):
    """Create a collision-safe filename using parent prefix."""
    prefix = rel_parent.replace("/", "_").replace("\\", "_")
    if prefix in ("", "."):
        return img.name
    return f"{prefix}_{img.name}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def stage_pair(img: Path, lbl: Path, out_images: Path, out_labels: Path, new_img_name: str, copy: bool):
    dst_img = out_images / new_img_name
    dst_lbl = out_labels / (Path(new_img_name).stem + LBL_EXT)
    if copy:
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl, dst_lbl)
    else:
        if not dst_img.exists():
            os.symlink(img, dst_img)
        if not dst_lbl.exists():
            os.symlink(lbl, dst_lbl)

def write_list(paths, out_file: Path):
    out_file.write_text("\n".join(str(p) for p in paths) + ("\n" if paths else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, help="Root with many nested folders (e.g., pt1)")
    ap.add_argument("--out-root", required=True, help="Output dataset folder (e.g., /.../datasets/dataset_pt1)")
    ap.add_argument("--valid-ratio", type=float, default=0.1, help="Fraction for valid split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()

    pairs = find_pairs(in_root)
    if not pairs:
        raise SystemExit("No (image,label) pairs found. Check your input root.")

    random.Random(args.seed).shuffle(pairs)
    n = len(pairs)
    n_valid = int(n * args.valid_ratio)
    valid_pairs = pairs[:n_valid]
    train_pairs = pairs[n_valid:]

    # Prepare dirs
    train_img = out_root / "train" / "images"; ensure_dir(train_img)
    train_lbl = out_root / "train" / "labels"; ensure_dir(train_lbl)
    valid_img = out_root / "valid" / "images"; ensure_dir(valid_img)
    valid_lbl = out_root / "valid" / "labels"; ensure_dir(valid_lbl)

    # Stage files
    for split_pairs, img_dir, lbl_dir in (
        (train_pairs, train_img, train_lbl),
        (valid_pairs, valid_img, valid_lbl),
    ):
        for img, lbl, rel_parent in split_pairs:
            new_name = safe_name(img, rel_parent)
            stage_pair(img, lbl, img_dir, lbl_dir, new_name, copy=args.copy)

    # Build absolute image lists
    def collect_abs_images(dir_: Path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files += [Path(p).resolve() for p in glob.glob(str(dir_ / ext))]
        return sorted(files)

    train_list = collect_abs_images(train_img)
    valid_list = collect_abs_images(valid_img)
    write_list(train_list, out_root / "train.txt")
    write_list(valid_list, out_root / "valid.txt")

    print(f"[done] Train: {len(train_list)} images -> {out_root/'train.txt'}")
    print(f"[done] Valid: {len(valid_list)} images -> {out_root/'valid.txt'}")
    print(f"Output dataset root: {out_root}")

if __name__ == "__main__":
    main()
