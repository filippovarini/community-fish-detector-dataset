"""
Tassie BRUV (UNSW Sydney / CSIRO Data61)
Source: https://datadryad.org/dataset/doi:10.5061/dryad.sbcc2frf7 (Tassie_BRUV.zip, 73.8 GB)
Paper: Maslen et al. (2025), Ecology and Evolution 15(8), e71996, https://doi.org/10.1002/ece3.71996
Download: only training_data_species_grouped (1,912 frames + YOLO labels, ~0.2 GB) and meta_data
    are used; the rest of the zip is the 28 source videos and movement-augmented copies of the
    frames. With Dryad API credentials (DRYAD_CLIENT_ID + DRYAD_CLIENT_SECRET, or DRYAD_TOKEN, in
    the environment or in .env) these folders are extracted straight from the remote zip with
    HTTP Range requests. Otherwise download Tassie_BRUV.zip manually to data/raw/tassie_bruv/
    and rerun.
Split logic: By site, read from the frame name (<date>_BRUV[S]_<site>_...): Sandy Bay (SB,
    17 videos) -> train; North West Bay (NWB, 7 videos) and Battery Point (BP, 4 videos) -> val.
    The authors' train/test/valid split is random over frames, so frames of one video are in
    all three. Frames are 50 random + up to 100 extra "rare fish" frames per video, so frames
    of one video can be close in time.
Category mapping (training_data_species_grouped labels; the species in each group are listed in
    meta_data/species_annotation_counts.csv):
  fish: all 8 classes - Carcharhiniformes (draughtboard and gummy sharks), Chyrosophyrs auratus
        (snapper), Moridae (morid cods), Perciformes_sandy (King George whiting, gobies),
        Perciformes_silver (Australian salmon, trevally), Ray (stingrays, eagle ray, skates),
        Scorpaeniformes (southern sand flathead, ocean perch), Tetradontiformes (leatherjackets,
        pufferfish)
  non-fish: none
  discarded: none
"""

import json
import re
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    HttpRangeFile,
    get_dryad_credentials_help,
    get_dryad_download_request,
    has_dryad_credentials,
    read_dryad_credentials,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "tassie_bruv"
DATA_PAGE_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.sbcc2frf7"
DRYAD_FILE_ID = 4431535  # Tassie_BRUV.zip

# Every source class is a fish (species grouped into higher taxa by the authors)
CATEGORIES_FILTER = {
    "Carcharhiniformes": "fish",
    "Chyrosophyrs auratus": "fish",
    "Moridae": "fish",
    "Perciformes_sandy": "fish",
    "Perciformes_silver": "fish",
    "Ray": "fish",
    "Scorpaeniformes": "fish",
    "Tetradontiformes": "fish",
}
SOURCE_SPLITS = ["train", "valid", "test"]
TRAIN_SITES = {"SB"}  # Sandy Bay; North West Bay (NWB) and Battery Point (BP) -> val
SITE_PATTERN = re.compile(r"\d{8}_BRUVS?_(NWB|SB|BP)_")

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_download_path / "Tassie_BRUV.zip"
source_root = raw_download_path / "Tassie_BRUV"
training_data_path = source_root / "training_data_species_grouped"
extraction_done_path = source_root / ".extraction_complete"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "tassie_bruv_mapped_annotations.json"


def is_needed_member(name: str) -> bool:
    if name.startswith("Tassie_BRUV/meta_data/") or name == "Tassie_BRUV/training_data_species_grouped/data.yaml":
        return True
    return re.fullmatch(
        r"Tassie_BRUV/training_data_species_grouped/(images/\w+/[^/]+\.jpg|labels/\w+/[^/]+\.txt)", name
    ) is not None


def download_data():
    """Extract the labelled frames from the local zip if there is one, otherwise from the remote zip"""
    if extraction_done_path.exists():
        print(f"Data already extracted: {source_root}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    credentials = read_dryad_credentials()

    if zip_path.exists():
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(
                f"{zip_path} exists but is not a complete zip. If a manual download is still "
                "running, wait for it to finish; otherwise delete the file and rerun."
            )
        print(f"Extracting the labelled frames from the local zip {zip_path}")
        zip_file = open(zip_path, "rb")
    elif has_dryad_credentials(credentials):
        print(f"Extracting the labelled frames from the remote zip (Dryad file {DRYAD_FILE_ID})")
        zip_file = HttpRangeFile(get_dryad_download_request(DRYAD_FILE_ID, credentials))
    else:
        raise RuntimeError(get_dryad_credentials_help(
            DATA_PAGE_URL, zip_path, "the 0.2 GB training_data_species_grouped folder"
        ))

    with zip_file, zipfile.ZipFile(zip_file) as zf:
        members = sorted(
            (m for m in zf.infolist() if not m.is_dir() and is_needed_member(m.filename)),
            key=lambda m: m.header_offset,  # archive order: one Range request per block
        )
        for member in tqdm(members, desc="Extracting labelled frames"):
            zf.extract(member, raw_download_path)

    extraction_done_path.touch()
    print(f"Extracted {len(members)} files to {source_root}")


def build_coco():
    """
    COCO file over all labelled frames (YOLO labels converted to absolute xywh boxes). Each image
    records the authors' split in "source_split", and its video and site.
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    with open(training_data_path / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]

    images, annotations = [], []
    clipped = 0
    for source_split in SOURCE_SPLITS:
        image_paths = sorted((training_data_path / "images" / source_split).glob("*.jpg"))
        for image_path in tqdm(image_paths, desc=source_split):
            label_path = training_data_path / "labels" / source_split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"No labels for {image_path}")

            shutil.copy2(image_path, coco_images_path / image_path.name)
            with Image.open(image_path) as im:
                width, height = im.size
            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "source_split": source_split,
                "video": image_path.stem.rsplit("_", 1)[0],
                "site": SITE_PATTERN.match(image_path.name).group(1),
            })

            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                class_id, xc, yc, w, h = line.split()
                xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
                x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
                x1, y1 = min(float(width), xc + w / 2), min(float(height), yc + h / 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                if (x1 - x0, y1 - y0) != (w, h):
                    clipped += 1
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                })

    with open(coco_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)} ({clipped} clipped to the image)")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    build_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    # By site: Sandy Bay -> train, North West Bay and Battery Point -> val
    with open(mapped_annotations_path) as f:
        images = json.load(f)["images"]

    train_images = {image["file_name"] for image in images if image["site"] in TRAIN_SITES}

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).name in train_images
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
