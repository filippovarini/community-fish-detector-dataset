"""
FjordFish
Source: https://zenodo.org/records/17950781 (FjordFish.zip)
Split logic: By blocks of 100 consecutive image ids (`<id> // 100`, 31 blocks). Images are
    named 1.jpg ... 3002.jpg and numbered in source order: consecutive ids are frames of the
    same BRUV/mesocosm video or photos of the same diver series. The authors' own
    train/val folders are a random split by image (neighbouring frames end up in both), so
    they are ignored. No video/site identifier is published.
Category mapping:
  fish: ballan, cuckoo_female, cuckoo_male, corkwing_female, corkwing_male, goldsinny,
        rock_cook, labrids_unknown, cod, cod_yoy, haddock, pollack, poor_cod, saithe,
        whiting, gadines_unknown, spiny_dogfish, sharks_unknown, ling, two_spot_goby,
        black_goby, sand_goby
  non-fish: brown_crab, green_crab, hermit (hermit crab), lobster, decapods_unknown (all crabs
        on inspection)
  discard: unknown (listed in data.yaml, no boxes)
"""

import json
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    extract_downloaded_file,
    CompressionType,
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "fjordfish"
DATA_URL = "https://zenodo.org/records/17950781/files/FjordFish.zip?download=1"
# YOLO class ids 0-27 in data.yaml order; every class except "unknown" (27, no boxes) is kept
CLASS_NAMES = [
    "ballan", "cuckoo_female", "cuckoo_male", "corkwing_female", "corkwing_male",
    "goldsinny", "rock_cook", "labrids_unknown", "cod", "cod_yoy", "haddock", "pollack",
    "poor_cod", "saithe", "whiting", "gadines_unknown", "spiny_dogfish", "sharks_unknown",
    "brown_crab", "green_crab", "hermit", "lobster", "decapods_unknown", "ling",
    "two_spot_goby", "black_goby", "sand_goby", "unknown",
]
CATEGORIES_FILTER = {
    "ballan": "fish",  # Labrus bergylta
    "cuckoo_female": "fish",  # Labrus mixtus
    "cuckoo_male": "fish",  # Labrus mixtus
    "corkwing_female": "fish",  # Symphodus melops
    "corkwing_male": "fish",  # Symphodus melops
    "goldsinny": "fish",  # Ctenolabrus rupestris
    "rock_cook": "fish",  # Centrolabrus exoletus
    "labrids_unknown": "fish",  # Labridae sp.
    "cod": "fish",  # Gadus morhua
    "cod_yoy": "fish",  # Gadus morhua, young of the year
    "haddock": "fish",  # Melanogrammus aeglefinus
    "pollack": "fish",  # Pollachius pollachius
    "poor_cod": "fish",  # Trisopterus minutus
    "saithe": "fish",  # Pollachius virens
    "whiting": "fish",  # Merlangius merlangus
    "gadines_unknown": "fish",  # Gadinae sp.
    "spiny_dogfish": "fish",  # Squalus acanthias
    "sharks_unknown": "fish",  # Squalidae sp. (no boxes)
    "brown_crab": "non-fish",  # Cancer pagurus
    "green_crab": "non-fish",  # Carcinus maenas
    "hermit": "non-fish",  # Paguridae sp.
    "lobster": "non-fish",  # Homarus gammarus
    "decapods_unknown": "non-fish",  # Decapoda sp. (crabs)
    "ling": "fish",  # Molva molva
    "two_spot_goby": "fish",  # Pomatoschistus flavescens
    "black_goby": "fish",  # Gobius niger
    "sand_goby": "fish",  # Pomatoschistus minutus
}
ID_BLOCK_SIZE = 100  # Images whose ids fall in the same block go to the same split

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / "FjordFish"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "fjordfish_mapped_annotations.json"


def download_data():
    if raw_data_dir.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / "FjordFish.zip"
    download_file_resumable(DATA_URL, zip_path)
    extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)


def yolo_to_coco():
    """
    Convert the YOLO labels of both source folders (train, val) into one COCO dataset.
    Labels are matched to images by file stem across folders: 2515.jpg is in images/train
    but its label is in labels/val.
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    image_paths = {p.stem: p for p in (raw_data_dir / "images").glob("*/*.jpg")}
    label_paths = {p.stem: p for p in (raw_data_dir / "labels").glob("*/*.txt")}
    missing = set(label_paths) - set(image_paths)
    assert not missing, f"Labels without image: {sorted(missing)[:5]}"

    images = []
    annotations = []
    for stem in tqdm(sorted(image_paths, key=int)):
        image_path = image_paths[stem]
        with Image.open(image_path) as im:
            width, height = im.size
        shutil.copy2(image_path, coco_images_path / image_path.name)
        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        if stem not in label_paths:
            continue
        for line in label_paths[stem].read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            # YOLO format: class_id cx cy w h (normalized)
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            bbox_w, bbox_h = w * width, h * height
            bbox_x, bbox_y = cx * width - bbox_w / 2, cy * height - bbox_h / 2
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)],
                "area": round(bbox_w * bbox_h, 2),
                "iscrowd": 0,
            })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, start=1)],
    }
    with open(coco_annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


def get_id_block(image_name: str) -> int:
    """fjordfish_1234.jpg -> 12"""
    return int(Path(image_name).stem.removeprefix(f"{DATASET_SHORTNAME}_")) // ID_BLOCK_SIZE


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    yolo_to_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    # Split by id block, reproducible: sorted blocks + fixed random_state
    with open(mapped_annotations_path) as f:
        image_names = [image["file_name"] for image in json.load(f)["images"]]
    train_blocks, _ = train_test_split(
        sorted({get_id_block(name) for name in image_names}),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    train_blocks = set(train_blocks)

    should_the_image_be_included_in_train_set = (
        lambda image_path: get_id_block(Path(image_path).name) in train_blocks
    )

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
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
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
