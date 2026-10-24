"""
RUOD-R (re-annotated Real-world Underwater Object Detection)
Source: annotations https://github.com/RSSL-MTU/RUOD-R (COCO, commit 988939f, "RUOD-R/COCO"),
    images https://github.com/xiaoDetection/RUOD (release RUOD.tar.partaa + RUOD.tar.partab)
Split logic: By source group, not the authors' train/test split. Photos are numbered in blocks from
    the same video, and the authors' random split puts neighbouring, near-identical frames on both
    sides. Photos are grouped (union-find) when they are consecutive file numbers of the same size,
    near-identical (normalised 32x18 grayscale thumbnails with correlation >= 0.95 at the same
    size, >= 0.98 at different sizes) or byte-identical. Whole groups are shuffled (random_state)
    and moved to val until val holds 20% of the photos.
Category mapping:
  fish: fish (except the 89 boxes on dolphins, see ruod_r_dolphins.json)
  non-fish: echinus (sea urchin), holothurian (sea cucumber), starfish, cuttlefish, turtle,
    jellyfish, and dolphin (fish boxes relabelled by this script)
  discard: scallop (a shell), corals, diver

The 14,000 RUOD photos come from many sources (seabed-harvesting video frames, diving videos,
aquariums, web photos) at 3,048 sizes. Of each group of byte-identical photos (756 groups, 1,643
photos; the copies were annotated separately and often differently) only the copy with the most
boxes is kept. Photo 008431 is dropped: its file is stored rotated and none of its 5 boxes fits any
orientation. Boxes are clipped to the image; boxes 1 pixel or less wide or high are dropped. Photos
without boxes are kept. Each COCO image keeps its RUOD split (ruod_split) and source group.
"""

import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "ruod_r"
ANNOTATIONS_COMMIT = "988939f93734b378f82d15a9a2f7371dae475363"
ANNOTATION_URLS = {
    split: f"https://raw.githubusercontent.com/RSSL-MTU/RUOD-R/{ANNOTATIONS_COMMIT}"
    f"/Annotations/RUOD-R/COCO/instances_{split}.json"
    for split in ("train", "test")
}
ANNOTATION_MD5 = {"train": "c2e309d2797e3aed2256d7bdf94abb85", "test": "de4bc7a49968adb963509f20ff34f3ae"}
IMAGE_PARTS = {
    "RUOD.tar.partaa": "3da9cd94e67b185403d0abaf019f8053",
    "RUOD.tar.partab": "f7fd200ba6e8a508f97d9283adadb098",
}
IMAGE_PARTS_URL = "https://github.com/xiaoDetection/RUOD/releases/download/untagged-4f7c7ab75187d68b6449/"

# Every source category must be listed here explicitly; unlisted categories are discarded.
CATEGORIES_FILTER = {
    "fish": "fish",
    "echinus": "non-fish",  # sea urchin
    "holothurian": "non-fish",  # sea cucumber
    "starfish": "non-fish",
    "cuttlefish": "non-fish",
    "turtle": "non-fish",
    "jellyfish": "non-fish",
    "dolphin": "non-fish",  # added by this script: RUOD-R labels the dolphins "fish"
}
# Discarded: scallop (a shell, does not count as free-moving), corals, diver

# The 89 "fish" boxes that are bottlenose dolphins ({file name: [[x, y, w, h], ...]}, boxes
# after clipping), found by checking every fish box at least 10% of the image in the diving videos
# with dolphins, and every grey fish box at least 15% of the image in the whole dataset
DOLPHINS_PATH = Path(__file__).with_name("ruod_r_dolphins.json")
DROPPED_PHOTOS = {"008431.jpg"}  # stored rotated; none of its boxes fits any orientation
KEEP_ONE_COPY_OF_IDENTICAL_PHOTOS = True
SAME_SIZE_THRESHOLD = 0.95
OTHER_SIZE_THRESHOLD = 0.98
MIN_BOX_SIZE = 1  # pixels; boxes this narrow or low after clipping are dropped

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
images_root = raw_data_dir / "RUOD" / "RUOD_pic"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "ruod_r_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name
groups_path = processing_dir / "source_groups.json"


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_checked(url: str, path: Path, expected_md5: str):
    download_file_resumable(url, path)
    actual = md5sum(path)
    if actual != expected_md5:
        raise ValueError(f"MD5 mismatch for {path}: expected {expected_md5}, got {actual}. Delete it and rerun.")


def download_data():
    """Download the RUOD-R annotations and the RUOD images (two tar parts), and extract the photos."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    for split, url in ANNOTATION_URLS.items():
        download_checked(url, raw_data_dir / f"ruod_r_instances_{split}.json", ANNOTATION_MD5[split])

    extracted_marker = raw_data_dir / ".images_extracted"
    if extracted_marker.exists():
        print(f"Images already extracted at {images_root}")
        return
    for name, md5 in IMAGE_PARTS.items():
        download_checked(IMAGE_PARTS_URL + name, raw_data_dir / name, md5)

    # The archive is split in two: stream both parts into tar, extracting only the photos
    print("Extracting RUOD/RUOD_pic from the image archive...")
    parts = [str(raw_data_dir / name) for name in IMAGE_PARTS]
    cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xf", "-", "-C", str(raw_data_dir), "./RUOD/RUOD_pic"], stdin=cat.stdout, check=True)
    cat.stdout.close()
    if cat.wait() != 0:
        raise RuntimeError("Reading the image archive parts failed")
    extracted_marker.touch()


def load_ruod_r() -> tuple[list[dict], dict[tuple[str, int], list[dict]], dict[int, str]]:
    """Photos of both RUOD splits (with their split), their boxes, and the category names."""
    photos, boxes, names = [], defaultdict(list), {}
    for split in ("train", "test"):
        coco = json.loads((raw_data_dir / f"ruod_r_instances_{split}.json").read_text())
        names.update({c["id"]: c["name"] for c in coco["categories"]})
        photos += [{**image, "ruod_split": split} for image in coco["images"]]
        for a in coco["annotations"]:
            boxes[(split, a["image_id"])].append(a)
    if len(photos) != 14000:
        raise ValueError(f"Expected 14,000 RUOD photos, found {len(photos)}")
    return photos, boxes, names


def photo_path(photo: dict) -> Path:
    return images_root / photo["ruod_split"] / photo["file_name"]


def find_source_groups(photos: list[dict]) -> tuple[dict[str, int], dict[str, str]]:
    """
    {file name: source group} for every photo (union-find over consecutive file numbers of the
    same size, near-identical thumbnails and byte-identical files), and {file name: MD5}.
    """
    photos = sorted(photos, key=lambda p: p["file_name"])
    n = len(photos)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    sizes, thumbs, digests = [], [], defaultdict(list)
    for i, photo in enumerate(tqdm(photos, desc="Comparing photos")):
        path = photo_path(photo)
        digests[md5sum(path)].append(i)
        with Image.open(path) as image:
            sizes.append(image.size)
            image.draft("RGB", (128, 128))
            thumb = np.asarray(image.convert("L").resize((32, 18), Image.BILINEAR), dtype=np.float32)
        thumbs.append(((thumb - thumb.mean()) / (thumb.std() + 1e-6)).ravel())
    thumbs = np.stack(thumbs)
    numbers = [int(Path(p["file_name"]).stem) for p in photos]

    for i in range(1, n):
        if sizes[i] == sizes[i - 1] and numbers[i] == numbers[i - 1] + 1:
            union(i, i - 1)
    for start in range(0, n, 2000):
        correlation = thumbs[start:start + 2000] @ thumbs.T / thumbs.shape[1]
        rows, cols = np.where(correlation >= min(SAME_SIZE_THRESHOLD, OTHER_SIZE_THRESHOLD))
        for i, j in zip(rows + start, cols):
            if i < j:
                threshold = SAME_SIZE_THRESHOLD if sizes[i] == sizes[j] else OTHER_SIZE_THRESHOLD
                if correlation[i - start, j] >= threshold:
                    union(i, j)
    for copies in digests.values():
        for i in copies[1:]:
            union(i, copies[0])

    roots = {}
    groups = {p["file_name"]: roots.setdefault(find(i), len(roots)) for i, p in enumerate(photos)}
    md5s = {photos[i]["file_name"]: digest for digest, copies in digests.items() for i in copies}
    return groups, md5s


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / union if union > 0 else 0.0


def clip_box(bbox: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 - x0 <= MIN_BOX_SIZE or y1 - y0 <= MIN_BOX_SIZE:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def convert_ruod_r_to_coco():
    """
    One COCO file for both RUOD splits with the source categories, copying the kept photos with
    the dataset prefix, and the source groups for the split.
    """
    if source_annotations_path.exists() and groups_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    photos, boxes, names = load_ruod_r()
    photos = [p for p in photos if p["file_name"] not in DROPPED_PHOTOS]
    groups, md5s = find_source_groups(photos)

    kept, dropped_copies = photos, 0
    if KEEP_ONE_COPY_OF_IDENTICAL_PHOTOS:
        by_digest = defaultdict(list)
        for p in photos:
            by_digest[md5s[p["file_name"]]].append(p)
        kept = []
        for copies in by_digest.values():
            # The copy with the most boxes; ties go to the lowest file number
            copies.sort(key=lambda p: (-len(boxes[(p["ruod_split"], p["id"])]), p["file_name"]))
            kept.append(copies[0])
            dropped_copies += len(copies) - 1
    kept.sort(key=lambda p: p["file_name"])

    images_path.mkdir(parents=True, exist_ok=True)
    if len(names) != 10 or min(names) != 1:
        raise ValueError(f"Unexpected categories: {names}")
    category_ids = {name: cid for cid, name in names.items()}
    category_ids["dolphin"] = max(names) + 1  # fish boxes on dolphins, relabelled
    coco = {"images": [], "annotations": [],
            "categories": [{"id": cid, "name": name} for name, cid in sorted(category_ids.items(), key=lambda kv: kv[1])]}
    dolphins = json.loads(DOLPHINS_PATH.read_text())
    dropped_boxes, relabelled = Counter(), 0
    for image_id, photo in enumerate(tqdm(kept, desc="Copying photos"), start=1):
        file_name = f"{DATASET_SHORTNAME}_{photo['file_name']}"
        shutil.copy2(photo_path(photo), images_path / file_name)
        coco["images"].append({
            "id": image_id, "file_name": file_name, "width": photo["width"], "height": photo["height"],
            "ruod_split": photo["ruod_split"], "source_group": groups[photo["file_name"]],
        })
        for a in boxes[(photo["ruod_split"], photo["id"])]:
            bbox = clip_box(a["bbox"], photo["width"], photo["height"])
            if bbox is None:
                dropped_boxes[names[a["category_id"]]] += 1
                continue
            name = names[a["category_id"]]
            if name == "fish" and any(box_iou(bbox, d) >= 0.95 for d in dolphins.get(photo["file_name"], [])):
                name, relabelled = "dolphin", relabelled + 1
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": image_id,
                "category_id": category_ids[name], "bbox": bbox,
                "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
    expected = sum(len(v) for k, v in dolphins.items() if k in {p["file_name"] for p in kept})
    if relabelled != expected:
        raise ValueError(f"Relabelled {relabelled} dolphin boxes, expected {expected}")

    print(f"Photos: {len(coco['images'])} kept ({dropped_copies} identical copies and "
          f"{len(DROPPED_PHOTOS)} misannotated photo dropped), boxes: {len(coco['annotations'])}, "
          f"boxes dropped as too small: {dict(dropped_boxes)}, fish boxes relabelled as dolphin: {relabelled}")
    groups_path.write_text(json.dumps({f"{DATASET_SHORTNAME}_{k}": v for k, v in groups.items()}))
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    partial.write_text(json.dumps(coco))
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_ruod_r_to_coco()
    map_annotations_to_fish_and_non_fish(source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path)


def get_validation_groups() -> set[int]:
    """Whole source groups, in a fixed random order, until val holds 20% of the photos."""
    with open(mapped_annotations_path) as f:
        images = json.load(f)["images"]
    group_sizes = Counter(image["source_group"] for image in images)
    order = sorted(group_sizes)
    random.Random(settings.random_state).shuffle(order)
    target = settings.train_val_split_ratio * len(images)
    validation, size = set(), 0
    for group in order:
        if size >= target:
            break
        if size + group_sizes[group] <= target * 1.1:  # skip groups that would overshoot by much
            validation.add(group)
            size += group_sizes[group]
    print(f"Validation: {len(validation)} source groups, {size} photos ({size / len(images):.1%})")
    return validation


def dataset_splitting():
    groups = json.loads(groups_path.read_text())
    validation_groups = get_validation_groups()
    should_the_image_be_included_in_train_set = (
        lambda image_path: groups[Path(image_path).name] not in validation_groups
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
