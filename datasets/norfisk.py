"""
NorFisk Dataset — farmed salmonids and wild saithe at Norwegian fish farms
Source: https://doi.org/10.18710/H5G3K5 (DataverseNO)
Split logic: By blocks of 100 image numbers within each species. Images are numbered in
    video order (1 frame per second from ~49 h of GoPro footage), but the video is not
    in the filename; neighbouring numbers look alike while numbers 100 apart are as
    different as random pairs, so blocks of 100 keep near-duplicate frames in one split.
    The split is done per species so both species are in train and val.
Category mapping:
  fish: salmonid (farmed salmon and trout, inside the net pens), saithe (wild, outside)
  non-fish: none
  discard: none
Known issue: each image is a crop around ONE fish (sizes vary, mostly 200-600 px wide)
    with exactly one box. Other fish visible in the crop are not annotated; this is
    frequent in the crowded salmonid crops. Included anyway at the user's request.
"""

import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    extract_downloaded_file,
    CompressionType,
    map_annotations_to_fish_and_non_fish,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)

DATASET_SHORTNAME = "norfisk"
# NorFisk_v1.0.zip (2.6 GB) on DataverseNO, file id 86889
DATA_URL = "https://dataverse.no/api/access/datafile/86889"
CATEGORIES_FILTER = {
    "salmonid": "fish",
    "saithe": "fish",
}
SPECIES_FOLDERS = ["saithe", "salmonid"]
IMAGE_BLOCK_SIZE = 100  # Images whose numbers fall in the same block go to the same split

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "norfisk_mapped_annotations.json"


def download_data():
    if all((raw_download_path / species / "img").exists() for species in SPECIES_FOLDERS):
        print(f"Data already downloaded: {raw_download_path}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / "NorFisk_v1.0.zip"
    download_file_resumable(DATA_URL, zip_path)
    # The zip holds saithe/{img,annotations}/ and salmonid/{img,annotations}/
    extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)


def norfisk_to_coco():
    """
    Each image <species>.<number>.png has an annotations/<species>.<number>.txt with
    lines "file_name,class,xmin,ymin,xmax,ymax,image_width,image_height".
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)
    category_ids = {name: i + 1 for i, name in enumerate(sorted(CATEGORIES_FILTER))}
    images, annotations = [], []

    for species in SPECIES_FOLDERS:
        species_dir = raw_download_path / species
        for label_file in tqdm(sorted((species_dir / "annotations").glob("*.txt")), desc=species):
            image_name = f"{label_file.stem}.png"
            shutil.copy2(species_dir / "img" / image_name, coco_images_path / image_name)

            image_id = len(images) + 1
            width = height = None
            for line in label_file.read_text().splitlines():
                if not line.strip():
                    continue
                file_name, class_name, *numbers = line.strip().split(",")
                assert file_name == image_name, f"{label_file} points to {file_name}"
                xmin, ymin, xmax, ymax, width, height = map(int, numbers)
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category_ids[class_name],
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "area": (xmax - xmin) * (ymax - ymin),
                        "iscrowd": 0,
                    }
                )
            images.append(
                {"id": image_id, "file_name": image_name, "width": width, "height": height}
            )

    categories = [{"id": i, "name": name} for name, i in category_ids.items()]
    with open(coco_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"COCO dataset: {len(images)} images, {len(annotations)} annotations")


def get_image_block(image_name: str) -> str:
    """norfisk_salmonid.001234.png -> salmonid_12"""
    species, number, _ = remove_dataset_shortname_prefix_from_image_filename(
        Path(image_name).name, DATASET_SHORTNAME
    ).split(".")
    return f"{species}_{int(number) // IMAGE_BLOCK_SIZE}"


def get_train_blocks() -> set[str]:
    with open(mapped_annotations_path) as f:
        blocks = sorted({get_image_block(image["file_name"]) for image in json.load(f)["images"]})
    train_blocks = []
    for species in SPECIES_FOLDERS:
        species_train_blocks, _ = train_test_split(
            [block for block in blocks if block.startswith(f"{species}_")],
            test_size=settings.train_val_split_ratio,
            random_state=settings.random_state,
        )
        train_blocks += species_train_blocks
    return set(train_blocks)


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    processing_dir.mkdir(parents=True, exist_ok=True)
    norfisk_to_coco()
    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )
    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_blocks = get_train_blocks()
    should_the_image_be_included_in_train_set = (
        lambda image_name: get_image_block(image_name) in train_blocks
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


if __name__ == "__main__":
    main()
