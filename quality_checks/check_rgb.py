"""
Checks that images in data/final are in RGB order (not BGR).

Underwater images are usually blue/green dominant. An image saved with swapped
channels becomes red/orange dominant, so for each dataset we sample images,
compute the mean R, G, B, and flag datasets where many images have R > B.
Some datasets are naturally reddish (aquariums, rivers, turbid water), so
flagged datasets should be confirmed by eye with the saved contact sheets.

Also reports images that fail to open or are not 3-channel RGB.

Outputs (in data/quality/rgb/):
  - rgb_summary.csv: one row per dataset
  - rgb_images.csv: one row per sampled image
  - <dataset>.jpg: contact sheet of the sampled images

Usage:
  python -m quality_checks.check_rgb [--samples 100] [--datasets deepfish fathomnet]
"""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from datasets.settings import Settings

settings = Settings()

OUTPUT_DIR = settings.base_dir / "quality" / "rgb"
THUMB_SIZE = 128
# A dataset is flagged if more than this share of sampled images has mean R > mean B
RED_DOMINANT_FLAG_RATIO = 0.5


def list_datasets():
    """Group train/val folders in data/final by dataset name."""
    datasets = {}
    for folder in sorted(settings.processed_dir.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        for suffix in (settings.train_dataset_suffix, settings.val_dataset_suffix):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        datasets.setdefault(name, []).append(folder / settings.images_folder_name)
    return datasets


def sample_images(image_dirs, n):
    paths = []
    for image_dir in image_dirs:
        if image_dir.exists():
            paths += [image_dir / f for f in os.listdir(image_dir)]
    random.Random(settings.random_state).shuffle(paths)
    return paths[:n]


def analyse_image(path):
    row = {"path": str(path), "mode": None, "error": None}
    try:
        with Image.open(path) as img:
            row["mode"] = img.mode
            img.draft("RGB", (512, 512))  # fast JPEG decoding at reduced size
            rgb = img.convert("RGB")
            rgb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        mean_r, mean_g, mean_b = np.asarray(rgb, dtype=np.float32).reshape(-1, 3).mean(0)
        row.update(mean_r=mean_r, mean_g=mean_g, mean_b=mean_b, red_dominant=mean_r > mean_b)
        return row, rgb
    except Exception as e:
        row["error"] = repr(e)
        return row, None


def save_contact_sheet(thumbs, out_path, cols=10):
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * THUMB_SIZE, rows * THUMB_SIZE))
    for i, thumb in enumerate(thumbs):
        x, y = (i % cols) * THUMB_SIZE, (i // cols) * THUMB_SIZE
        sheet.paste(thumb, (x + (THUMB_SIZE - thumb.width) // 2, y + (THUMB_SIZE - thumb.height) // 2))
    sheet.save(out_path, quality=85)


def check_dataset(name, image_dirs, n_samples):
    paths = sample_images(image_dirs, n_samples)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(analyse_image, paths))

    image_rows = [{"dataset": name, **row} for row, _ in results]
    thumbs = [thumb for _, thumb in results if thumb is not None]
    if thumbs:
        save_contact_sheet(thumbs, OUTPUT_DIR / f"{name}.jpg")

    df = pd.DataFrame(image_rows)
    ok = df[df["error"].isna()]
    summary = {
        "dataset": name,
        "sampled": len(df),
        "errors": int(df["error"].notna().sum()),
        "non_rgb_mode": int((ok["mode"] != "RGB").sum()),
        "mean_r": ok["mean_r"].mean() if len(ok) else np.nan,
        "mean_g": ok["mean_g"].mean() if len(ok) else np.nan,
        "mean_b": ok["mean_b"].mean() if len(ok) else np.nan,
        "red_dominant_ratio": ok["red_dominant"].mean() if len(ok) else np.nan,
    }
    summary["flagged"] = bool(summary["red_dominant_ratio"] > RED_DOMINANT_FLAG_RATIO)
    return summary, image_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--datasets", nargs="*", help="Only check these datasets")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = list_datasets()
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    summaries, all_image_rows = [], []
    for name, image_dirs in datasets.items():
        summary, image_rows = check_dataset(name, image_dirs, args.samples)
        summaries.append(summary)
        all_image_rows += image_rows
        status = "FLAGGED" if summary["flagged"] else "ok"
        print(
            f"{name:25s} R={summary['mean_r']:6.1f} G={summary['mean_g']:6.1f} B={summary['mean_b']:6.1f} "
            f"red_dominant={summary['red_dominant_ratio']:.0%} errors={summary['errors']} "
            f"non_rgb={summary['non_rgb_mode']}  {status}"
        )

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "rgb_summary.csv", index=False)
    pd.DataFrame(all_image_rows).to_csv(OUTPUT_DIR / "rgb_images.csv", index=False)
    print(f"\nFlagged: {[s['dataset'] for s in summaries if s['flagged']]}")
    print(f"Results and contact sheets saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
