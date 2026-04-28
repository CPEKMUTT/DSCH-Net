import random
import shutil
from pathlib import Path

# -------------------
# Settings
# -------------------
SEED = 42          # set for reproducibility; change or set to None for different split
MOVE = False       # False = copy files, True = move files

# Source folders
SRC_GT = Path("./dataset/RICE/RICE2/label")
SRC_HAZY = Path("./dataset/RICE/RICE2/cloud")

# Destination structure
DST_TRAIN_GT  = Path("./dataset/RICE/RICE2/train/label")
DST_TEST_GT   = Path("./dataset/RICE/RICE2/test/label")
DST_TRAIN_HZY = Path("./dataset/RICE/RICE2/train/cloud")
DST_TEST_HZY  = Path("./dataset/RICE/RICE2/test/cloud")


N_TRAIN = 636
N_TEST  = 100

def ensure_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def list_pngs_map(folder: Path):
    # Map filename (case-sensitive) -> full path, only .png (case-insensitive suffix)
    return {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"}

def main():
    ensure_dirs(DST_TRAIN_GT, DST_TEST_GT, DST_TRAIN_HZY, DST_TEST_HZY)

    gt_map   = list_pngs_map(SRC_GT)
    hazy_map = list_pngs_map(SRC_HAZY)

    # Pair only files that exist in BOTH folders with the SAME filename
    common_names = sorted(set(gt_map.keys()) & set(hazy_map.keys()))
    total_pairs = len(common_names)

    if total_pairs < (N_TRAIN + N_TEST):
        raise ValueError(
            f"Found only {total_pairs} matched PNG pairs between GT/ and hazy/; "
            f"need {N_TRAIN + N_TEST}."
        )

    # Build list of (gt_path, hazy_path) pairs
    pairs = [(gt_map[name], hazy_map[name]) for name in common_names]

    # Random split of PAIRS
    if SEED is not None:
        random.seed(SEED)
    train_pairs = random.sample(pairs, N_TRAIN)
    train_set = set(train_pairs)
    test_pairs = [p for p in pairs if p not in train_set][:N_TEST]  # exact 100

    # Copy/Move helper
    op = shutil.move if MOVE else shutil.copy2
    for gt, hz in train_pairs:
        op(str(gt), str(DST_TRAIN_GT / gt.name))
        op(str(hz), str(DST_TRAIN_HZY / hz.name))
    for gt, hz in test_pairs:
        op(str(gt), str(DST_TEST_GT / gt.name))
        op(str(hz), str(DST_TEST_HZY / hz.name))

    action = "Moved" if MOVE else "Copied"
    print(f"{action} {len(train_pairs)} GT/hazy pairs to training/.")
    print(f"{action} {len(test_pairs)} GT/hazy pairs to testing.")
    # Optional: info about unmatched files (present in one folder only)
    extra_gt   = len(gt_map) - total_pairs
    extra_hazy = len(hazy_map) - total_pairs
    if extra_gt or extra_hazy:
        print(f"Note: {extra_gt} GT-only PNGs and {extra_hazy} hazy-only PNGs were ignored.")

if __name__ == "__main__":
    main()
