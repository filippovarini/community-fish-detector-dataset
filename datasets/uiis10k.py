"""
UIIS10K (Underwater Image Instance Segmentation, 10K)
Source: https://github.com/LiamLian0727/UIIS10K, data on Hugging Face
    (https://huggingface.co/datasets/LiamLian0727/UIIS10K, UIIS10K.zip)
Split logic: The UIIS10K split (train -> train, test -> val, 20%). The photos carry no location,
    video or source information and their file names are shuffled, so the official random split
    is kept, after removing copies of the same photo. Photos of the same shot that are not copies
    (for example two frames of one deep-sea video with a different object in view) are kept and put
    in the split of the one with the most boxes, so that no shot is in both splits.
Category mapping (source classes, with the boxes listed in uiis10k_relabels.json relabelled first):
  fish: fish
  non-fish: reptiles, arthropoda, and the relabelled marine mammals, birds, sea turtles and sea
    snakes, other reptiles, octopus/squid/cuttlefish, crustaceans, starfish, sea urchins, sea
    cucumbers and jellyfish
  discard: mollusk (every box is relabelled), corals, plants, ruins, garbage, robots, human, and
    the relabelled humans, shells, anemones, corals/sponges, nudibranchs, plants, objects and
    unclear boxes

UIIS10K (Li et al., 2025) gathers 10,048 web photos and photos of other public datasets (UIIS,
TrashCan, USOD10K, USIS10K, image enhancement sets) with instance masks for 10 classes. The classes
do not follow our definitions: "fish" covers "fish and other marine vertebrates" (dolphins, whales,
seals, penguins, crocodiles), "mollusk" holds anything from octopus, starfish and sea urchins to
clams, anemones, corals and stingrays, and "reptiles" and "arthropoda" hold some eels and other
fish. uiis10k_relabels.json gives a new label to 2,928 boxes. Every fish, mollusk, reptiles and
arthropoda box was labelled by Claude Haiku from a crop, and the boxes were then checked by eye:
every box Haiku did not call a fish (fish class), every mollusk box, and every reptiles and
arthropoda box. Boxes that could not be identified are labelled "unclear" and discarded.

Boxes are recomputed from the polygons (9 source boxes do not match their polygon, 2 of them are
empty) and clipped; boxes 1 pixel or less wide or high are dropped. A fish or non-fish instance
whose polygon has parts far apart (several fish joined into one instance) gets one box per group
of nearby parts. Of two boxes of the same output class with IoU >= 0.85 in a photo, one is dropped.

Duplicates: copies of the same photo (byte-identical, or normalised high-pass 64x48 grayscale
thumbnails with correlation >= 0.96, mostly the same photo at 640x480 and 320x240, or recoloured)
were annotated separately; of each group only the copy with the most fish and non-fish boxes is
kept. Photos that are also in RUOD (listed in uiis10k_in_ruod.json) are dropped: RUOD-R annotates
them more densely. Photos without boxes after mapping are kept.
"""

import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "uiis10k"
HF_REVISION = "5c4316368a8e420db754d1ab37c205a42985ba47"
ZIP_URL = f"https://huggingface.co/datasets/LiamLian0727/UIIS10K/resolve/{HF_REVISION}/UIIS10K.zip"
ZIP_MD5 = "d8ffd4d67c16c5537049460419bc0bf7"

# Every source category must be listed here explicitly; unlisted categories are discarded.
CATEGORIES_FILTER = {
    "fish": "fish",
    "reptiles": "non-fish",  # sea turtles, sea snakes
    "arthropoda": "non-fish",  # crabs, shrimp, lobsters
    # Labels given by uiis10k_relabels.json
    "marine mammal": "non-fish",  # dolphins, whales, belugas, orcas, seals, sea lions, walruses, manatees, polar bears
    "bird": "non-fish",  # penguins
    "turtle or sea snake": "non-fish",
    "other reptile": "non-fish",  # crocodiles, marine iguanas
    "octopus, squid or cuttlefish": "non-fish",  # nautilus too
    "crustacean": "non-fish",
    "starfish": "non-fish",  # feather stars too
    "sea urchin": "non-fish",  # sand dollars too
    "sea cucumber": "non-fish",
    "jellyfish": "non-fish",
}
# Discarded: mollusk (every box is relabelled), corals, plants, ruins, garbage, robots, human, and the
# relabelled "human", "shell" (clams, snails, empty shells), "anemone", "coral or sponge",
# "nudibranch" (size doubtful), "plant", "object" and "unclear"

# Boxes whose source class is wrong: {file name: [[annotation id, source class, new label], ...]}
RELABELS_PATH = Path(__file__).with_name("uiis10k_relabels.json")
# UIIS10K photos that are also RUOD photos: {UIIS10K file name: RUOD file}
IN_RUOD_PATH = Path(__file__).with_name("uiis10k_in_ruod.json")
COPY_THRESHOLD = 0.96  # thumbnail correlation of two copies of one photo
SAME_SHOT_THRESHOLD = 0.9  # thumbnail correlation of two photos of one shot
MIN_BOX_SIZE = 1  # pixels
SAME_BOX_IOU = 0.85
PART_GAP = 0.5  # polygon parts further apart than this times the larger part are separate animals
SMALL_PART = 0.15  # groups of parts smaller than this fraction of the largest group get no box

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_data_dir / "UIIS10K.zip"
uiis_root = raw_data_dir / "UIIS10K"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "uiis10k_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download UIIS10K.zip from Hugging Face (pinned revision) and extract it."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extracted_marker = raw_data_dir / ".extracted"
    if extracted_marker.exists():
        print(f"UIIS10K already extracted at {uiis_root}")
        return
    download_file_resumable(ZIP_URL, zip_path)
    actual = md5sum(zip_path)
    if actual != ZIP_MD5:
        raise ValueError(f"MD5 mismatch for {zip_path}: expected {ZIP_MD5}, got {actual}. Delete it and rerun.")
    print("Extracting UIIS10K.zip...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_data_dir)
    extracted_marker.touch()


def load_uiis10k() -> tuple[list[dict], dict[str, list[dict]]]:
    """Photos of both UIIS10K splits (with their split) and their annotations with class names."""
    photos, annotations = [], defaultdict(list)
    for split in ("train", "test"):
        coco = json.loads((uiis_root / "annotations" / f"multiclass_{split}.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        files = {}
        for image in coco["images"]:
            photos.append({**image, "uiis_split": split})
            files[image["id"]] = image["file_name"]
        for a in coco["annotations"]:
            annotations[files[a["image_id"]]].append({**a, "name": names[a["category_id"]]})
    if len(photos) != 10048:
        raise ValueError(f"Expected 10,048 UIIS10K photos, found {len(photos)}")
    return photos, annotations


def apply_relabels(annotations: dict[str, list[dict]]) -> Counter:
    """Relabel the boxes listed in uiis10k_relabels.json, checking each one's source class."""
    relabels = json.loads(RELABELS_PATH.read_text())
    counts = Counter()
    for file_name, items in relabels.items():
        by_id = {a["id"]: a for a in annotations[file_name]}
        for annotation_id, source, label in items:
            a = by_id[annotation_id]
            if a["name"] != source:
                raise ValueError(f"{file_name} annotation {annotation_id} is {a['name']}, expected {source}")
            a["name"] = label
            counts[(source, label)] += 1
    return counts


def high_pass_thumbnail(path: Path) -> np.ndarray:
    """Normalised 64x48 grayscale thumbnail minus its blurred self (unit vector)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gray = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    detail = (detail - detail.mean()) / (detail.std() + 1e-6)
    return detail.ravel() / np.sqrt(detail.size)


def union_find_groups(n: int, pairs) -> list[list[int]]:
    """Groups of indices 0..n-1 joined by the given pairs."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def compare_photos(photos: list[dict]) -> tuple[list[str], np.ndarray]:
    """MD5 of every photo and the thumbnail correlation of every pair."""
    digests, thumbs = [], []
    for photo in tqdm(photos, desc="Comparing photos"):
        path = uiis_root / "img" / photo["file_name"]
        digests.append(md5sum(path))
        thumbs.append(high_pass_thumbnail(path))
    thumbs = np.stack(thumbs)
    return digests, thumbs @ thumbs.T


def part_boxes(segmentation: list[list[float]], width: int, height: int) -> list[list[float]]:
    """
    One box per group of nearby polygon parts: parts whose gap is at most PART_GAP times the larger
    part belong to the same animal. Groups much smaller than the largest one get no box.
    """
    parts = []
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) >= 3:
            x0, y0 = np.clip(points.min(0), 0, [width, height])
            x1, y1 = np.clip(points.max(0), 0, [width, height])
            parts.append([x0, y0, x1, y1])
    if not parts:
        return []
    pairs = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]
            gap = max(max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]), 0)
            size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
            if gap <= PART_GAP * size:
                pairs.append((i, j))
    boxes = []
    for members in union_find_groups(len(parts), pairs):
        x0, y0 = min(parts[i][0] for i in members), min(parts[i][1] for i in members)
        x1, y1 = max(parts[i][2] for i in members), max(parts[i][3] for i in members)
        boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    largest = max(w * h for _, _, w, h in boxes)
    return [b for b in boxes if b[2] * b[3] >= SMALL_PART * largest]


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / union if union > 0 else 0.0


def photo_boxes(photo: dict, annotations: list[dict], stats: Counter) -> list[tuple[str, list[float]]]:
    """(class name, box) for the annotations of one photo, with the box fixes applied."""
    boxes = []
    for a in annotations:
        if a["name"] in CATEGORIES_FILTER:
            parts = part_boxes(a["segmentation"], photo["width"], photo["height"])
            stats["instances split into several boxes"] += len(parts) > 1
        else:
            parts = part_boxes([sum(a["segmentation"], [])], photo["width"], photo["height"])
        for bbox in parts:
            if bbox[2] <= MIN_BOX_SIZE or bbox[3] <= MIN_BOX_SIZE:
                stats["boxes dropped as too small"] += 1
                continue
            target = CATEGORIES_FILTER.get(a["name"])
            if target and any(CATEGORIES_FILTER.get(name) == target and box_iou(bbox, other) >= SAME_BOX_IOU
                              for name, other in boxes):
                stats["boxes dropped as duplicates"] += 1
                continue
            boxes.append((a["name"], bbox))
    return boxes


def select_photos(photos: list[dict], boxes: dict[str, list]) -> tuple[list[dict], Counter]:
    """
    One copy of each photo (the one with the most fish and non-fish boxes, then the largest), each
    with the split it goes to: photos of the same shot go to the split of the one with the most boxes.
    """
    def rank(photo):
        kept_boxes = sum(name in CATEGORIES_FILTER for name, _ in boxes[photo["file_name"]])
        return -kept_boxes, -photo["width"] * photo["height"], photo["file_name"]

    digests, correlation = compare_photos(photos)
    stats = Counter()
    copy_pairs = list(zip(*np.where(np.triu(correlation, 1) >= COPY_THRESHOLD)))
    first_with_digest = {}
    for i, digest in enumerate(digests):
        if digest in first_with_digest:
            copy_pairs.append((i, first_with_digest[digest]))
        first_with_digest.setdefault(digest, i)
    kept = []
    for group in union_find_groups(len(photos), copy_pairs):
        copies = sorted(group, key=lambda i: rank(photos[i]))
        kept.append(copies[0])
        stats["copies dropped"] += len(copies) - 1
        stats["copy groups across the UIIS10K split"] += len({photos[i]["uiis_split"] for i in group}) > 1

    sub = correlation[np.ix_(kept, kept)]
    shot_pairs = list(zip(*np.where(np.triu(sub, 1) >= SAME_SHOT_THRESHOLD)))
    selected = []
    for group in union_find_groups(len(kept), shot_pairs):
        members = sorted((photos[kept[i]] for i in group), key=rank)
        for photo in members:
            selected.append({**photo, "split": members[0]["uiis_split"]})
            stats["photos moved to the split of their shot"] += photo["uiis_split"] != members[0]["uiis_split"]
    return selected, stats


def convert_uiis10k_to_coco():
    """One COCO file with the kept photos (copied with the dataset prefix) and relabelled classes."""
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    photos, annotations = load_uiis10k()
    relabel_counts = apply_relabels(annotations)
    in_ruod = json.loads(IN_RUOD_PATH.read_text())
    photos = [p for p in photos if p["file_name"] not in in_ruod]

    stats = Counter()
    boxes = {p["file_name"]: photo_boxes(p, annotations[p["file_name"]], stats) for p in photos}
    kept, selection_stats = select_photos(photos, boxes)
    kept.sort(key=lambda p: (p["uiis_split"] != "train", p["file_name"]))

    names = sorted({name for p in kept for name, _ in boxes[p["file_name"]]})
    category_ids = {name: i for i, name in enumerate(names, start=1)}
    coco = {"images": [], "annotations": [],
            "categories": [{"id": cid, "name": name} for name, cid in category_ids.items()]}
    images_path.mkdir(parents=True, exist_ok=True)
    for image_id, photo in enumerate(tqdm(kept, desc="Copying photos"), start=1):
        file_name = f"{DATASET_SHORTNAME}_{photo['file_name']}"
        shutil.copy2(uiis_root / "img" / photo["file_name"], images_path / file_name)
        coco["images"].append({"id": image_id, "file_name": file_name, "width": photo["width"],
                               "height": photo["height"], "uiis_split": photo["uiis_split"],
                               "split": photo["split"]})
        for name, bbox in boxes[photo["file_name"]]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                        "category_id": category_ids[name], "bbox": bbox,
                                        "area": bbox[2] * bbox[3], "iscrowd": 0})

    print(f"Photos: {len(kept)} kept, {len(in_ruod)} also in RUOD dropped, {dict(selection_stats)}")
    print(f"Relabelled boxes: {sum(relabel_counts.values())}")
    for (source, label), n in sorted(relabel_counts.items()):
        print(f"  {source} -> {label}: {n}")
    print(f"Box fixes: {dict(stats)}")
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    partial.write_text(json.dumps(coco))
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_uiis10k_to_coco()
    map_annotations_to_fish_and_non_fish(source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path)


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        split = {image["file_name"]: image["split"] for image in json.load(f)["images"]}
    should_the_image_be_included_in_train_set = lambda image_path: split[Path(image_path).name] == "train"

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
