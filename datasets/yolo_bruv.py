"""
YOLO-BRUV (baited remote underwater video at Scottish offshore wind farms)
Source: https://doi.org/10.5281/zenodo.19594599 (YOLO-BRUV_data.zip)
Split logic: By video. Every frame comes from one of 19 GoPro recordings (the 4 digits after
    "RGX" and the 2-digit chapter in the file name), each one camera deployment. Whole videos are
    shuffled (random_state) and moved to val until val holds 20% of the frames.
Category mapping (species labels of the raw EventMeasure exports):
  fish: dab/flounder, haddock, unidentified flatfish, gobies, short-spined sea scorpion, grey
    gurnard, small-spotted catshark, plaice, dragonet, cod, ling, whiting, flapper skate, dab,
    poor cod
  non-fish: hermit crab, common starfish, edible crab, brittle star, common whelk (alive,
    crawling), squid
  discard: "Botryllus spp", a marker the annotators put in open water to export empty frames

The frames were exported from the videos with EventMeasure (every 1000th frame, plus extra frames)
in 6 annotation groups (G1-G6), and every animal was marked with a box (top-left corner, width and
height in pixels). The raw exports are used, not the "Corrected" ones, which only regroup the
species. Frames:
  - 68 frames of G1 in which animals are marked with points instead of boxes (other animals in them
    are not marked) are dropped
  - 12 frames were exported into two or three groups and annotated separately: the copy with the
    most boxes is kept (copies with points lose)
  - frames with only the empty-frame marker (455) are kept as background
Boxes are clipped to the image; boxes 1 pixel or less wide or high are dropped. Each COCO image
keeps its video, chapter, frame number and annotation group.
"""

import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "yolo_bruv"
ZIP_URL = "https://zenodo.org/api/records/19594599/files/YOLO-BRUV_data.zip/content"
ZIP_MD5 = "11e83d8b5172b0528f7105da7bfe92a9"

# Every source category must be listed here explicitly; unlisted categories are discarded.
# Source categories are "Genus species" from the EventMeasure exports (the species alone when the
# genus is missing).
CATEGORIES_FILTER = {
    "Limanda-Platichthys limanda-flesus": "fish",  # dab or flounder
    "Melanogrammus aeglefinus": "fish",  # haddock
    "Pleuronectiformes spp": "fish",  # unidentified flatfish
    "minutus": "fish",  # gobies (Gobiidae, genus not given)
    "Myoxocephalus scorpius": "fish",  # short-spined sea scorpion
    "Eutrigla gurnardus": "fish",  # grey gurnard
    "Scyliorhinus canicula": "fish",  # small-spotted catshark
    "Pleuronectes platessa": "fish",  # plaice
    "Callionymus lyra": "fish",  # dragonet
    "Gadus morhua": "fish",  # cod
    "Molva molva": "fish",  # ling
    "Merlangius merlangus": "fish",  # whiting
    "Dipturus intermedius": "fish",  # flapper skate
    "Limanda limanda": "fish",  # dab
    "Trisopterus minutus": "fish",  # poor cod
    "Pagurus bernhardus": "non-fish",  # hermit crab
    "Asterias rubens": "non-fish",  # common starfish
    "Cancer pagurus": "non-fish",  # edible crab
    "Ophiura albida": "non-fish",  # brittle star
    "Buccinum undatum": "non-fish",  # common whelk, alive and crawling in all its frames
    "vulgaris": "non-fish",  # a squid (genus not given)
}
# Discarded: "Botryllus spp", the marker for empty frames (not the sea squirt)

GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6"]
FILE_NAME = re.compile(r"^\d+_RGX(?P<chapter>\d{2})(?P<video>\d{4})\.(?:MP4|mp4)_(?P<frame>\d+)\.jpg$")
MIN_BOX_SIZE = 1  # pixels

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_data_dir / "YOLO-BRUV_data.zip"
annotations_root = raw_data_dir / "YOLO-BRUV_data" / "data" / "Annotations"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "yolo_bruv_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download YOLO-BRUV_data.zip from Zenodo and extract the annotations and frames."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extracted_marker = raw_data_dir / ".extracted"
    if extracted_marker.exists():
        print(f"YOLO-BRUV already extracted at {annotations_root}")
        return
    download_file_resumable(ZIP_URL, zip_path)
    actual = md5sum(zip_path)
    if actual != ZIP_MD5:
        raise ValueError(f"MD5 mismatch for {zip_path}: expected {ZIP_MD5}, got {actual}. Delete it and rerun.")
    print("Extracting the annotations and frames...")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith("YOLO-BRUV_data/data/Annotations/")]
        z.extractall(raw_data_dir, members)
    extracted_marker.touch()


def source_category(genus, species) -> str:
    """'Genus species', or the species alone when the genus is missing or repeats it."""
    if pd.isna(genus) or genus == species:
        return species
    return f"{genus} {species}"


def load_annotations() -> pd.DataFrame:
    """All rows of the raw EventMeasure exports, with their group and source category."""
    tables = []
    for group in GROUPS:
        table = pd.read_csv(annotations_root / "EMOutputs" / "Raw" / f"{group}_Points.txt", sep="\t")
        tables.append(table.assign(group=group))
    rows = pd.concat(tables, ignore_index=True)
    rows["category"] = [source_category(g, s) for g, s in zip(rows.Genus, rows.Species)]
    rows["is_box"] = (rows.RectWidth > 0) & (rows.RectHeight > 0)
    if (rows.Number != 1).any():
        raise ValueError("Expected one animal per annotation row")
    return rows


def select_frames(rows: pd.DataFrame) -> tuple[dict[str, tuple[str, pd.DataFrame]], Counter]:
    """
    {file name: (group, rows)} for the frames kept: of the copies of a frame exported into several
    groups, the one without point annotations and with the most boxes; frames whose kept copy has
    point annotations (other than the empty-frame marker) are dropped.
    """
    copies = defaultdict(list)
    for group in GROUPS:
        for path in sorted((annotations_root / "Images" / group).glob("*.jpg")):
            copies[path.name].append(group)
    rows_by_frame = {key: table for key, table in rows.groupby(["group", "Filename"])}
    if set(rows.Filename) - set(copies):
        raise ValueError("Annotated frames missing from the image folders")

    stats = Counter()
    kept = {}
    for file_name, groups in copies.items():
        candidates = []
        for group in groups:
            frame_rows = rows_by_frame.get((group, file_name), rows.iloc[:0])
            animal_rows = frame_rows[frame_rows.category.isin(CATEGORIES_FILTER)]
            has_points = bool((~animal_rows.is_box).any())
            candidates.append((has_points, -int(animal_rows.is_box.sum()), group, frame_rows))
        candidates.sort(key=lambda c: c[:3])
        has_points, _, group, frame_rows = candidates[0]
        stats["copies dropped"] += len(groups) - 1
        if has_points:
            stats["frames with point annotations dropped"] += 1
            continue
        kept[file_name] = (group, frame_rows)
    return kept, stats


def clip_box(row, width: int, height: int) -> tuple[list[float] | None, bool]:
    """The box clipped to the image (None if too small after clipping), and whether it was clipped."""
    x, y, w, h = float(row.ImageCol), float(row.ImageRow), float(row.RectWidth), float(row.RectHeight)
    if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
        return ([x, y, w, h] if w > MIN_BOX_SIZE and h > MIN_BOX_SIZE else None), False
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 - x0 <= MIN_BOX_SIZE or y1 - y0 <= MIN_BOX_SIZE:
        return None, True
    return [x0, y0, x1 - x0, y1 - y0], True


def convert_yolo_bruv_to_coco():
    """One COCO file with the kept frames (copied with the dataset prefix) and source categories."""
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    rows = load_annotations()
    kept, stats = select_frames(rows)
    names = sorted(set(rows.category))
    category_ids = {name: i for i, name in enumerate(names, start=1)}
    coco = {"images": [], "annotations": [],
            "categories": [{"id": cid, "name": name} for name, cid in category_ids.items()]}
    images_path.mkdir(parents=True, exist_ok=True)
    for image_id, file_name in enumerate(tqdm(sorted(kept), desc="Copying frames"), start=1):
        group, frame_rows = kept[file_name]
        source = annotations_root / "Images" / group / file_name
        with Image.open(source) as image:
            width, height = image.size
        parts = FILE_NAME.match(file_name)
        new_name = f"{DATASET_SHORTNAME}_{file_name}"
        shutil.copy2(source, images_path / new_name)
        coco["images"].append({
            "id": image_id, "file_name": new_name, "width": width, "height": height,
            "video": parts["video"], "chapter": parts["chapter"], "frame": int(parts["frame"]), "group": group,
        })
        for row in frame_rows[frame_rows.is_box].itertuples():
            bbox, clipped = clip_box(row, width, height)
            stats["boxes reaching past the image edge (clipped)"] += clipped
            if bbox is None:
                stats["boxes dropped as too small"] += 1
                continue
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": image_id,
                "category_id": category_ids[row.category], "bbox": bbox,
                "area": bbox[2] * bbox[3], "iscrowd": 0,
            })

    print(f"Frames: {len(coco['images'])} kept, {dict(stats)}")
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    partial.write_text(json.dumps(coco))
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_yolo_bruv_to_coco()
    map_annotations_to_fish_and_non_fish(source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path)


def get_validation_videos(images: list[dict]) -> set[str]:
    """Whole videos, in a fixed random order, until val holds 20% of the frames."""
    video_sizes = Counter(image["video"] for image in images)
    order = sorted(video_sizes)
    random.Random(settings.random_state).shuffle(order)
    target = settings.train_val_split_ratio * len(images)
    validation, size = set(), 0
    for video in order:
        if size >= target:
            break
        if size + video_sizes[video] <= target * 1.1:  # skip videos that would overshoot by much
            validation.add(video)
            size += video_sizes[video]
    print(f"Validation videos: {sorted(validation)}, {size} frames ({size / len(images):.1%})")
    return validation


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        images = json.load(f)["images"]
    video = {image["file_name"]: image["video"] for image in images}
    validation_videos = get_validation_videos(images)
    should_the_image_be_included_in_train_set = (
        lambda image_path: video[Path(image_path).name] not in validation_videos
    )

    train_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    val_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        images_path,
        mapped_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
