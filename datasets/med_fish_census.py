"""
Mediterranean Fish Census (autoUC, Université Côte d'Azur)
Source: https://datadryad.org/dataset/doi:10.5061/dryad.f7m0cfz6f (autoUC_Data.zip, 121.7 GB)
Paper: Bürgi et al. (2025), Ecological Informatics 85, 102959, https://doi.org/10.1016/j.ecoinf.2024.102959
Download: only Data_T (training/test frames + labels, 4.4 GB) is used. With Dryad API
    credentials (DRYAD_CLIENT_ID + DRYAD_CLIENT_SECRET, or DRYAD_TOKEN, in the environment or
    in .env) Data_T is extracted straight from the remote zip with HTTP Range requests.
    Otherwise download autoUC_Data.zip manually to data/raw/med_fish_census/ and rerun.
Split logic: The authors' split by video (their testSet -> val, their trainSet -> train), except
    that GoPro recordings whose clips appear in both of their splits (e.g. GH010003 in test and
    GH010003_2 in train) go entirely to val. 63 frames are byte-identical in both of their splits
    (videos GH010143_1, GH010149, GH010211_2); only one copy is kept.
Category mapping (labels_allSpecies; labels_Reduced for the 118 train images that have no
    labels_allSpecies file):
  fish: the 40 species classes and "Other" (the rare species grouped together)
  non-fish: jellyfish (Pelagia noctiluca), seaurchin
  discarded: background (hard negatives: rocks, seagrass, wreck parts, blur), diver, drone (no
        boxes), fish (its 4 boxes are seagrass leaves)
Frames with no boxes left after mapping (mostly background/diver-only frames) are kept as
    background images.
"""

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    HttpRangeFile,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "med_fish_census"
DATA_PAGE_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.f7m0cfz6f"
DRYAD_FILE_ID = 3732300  # autoUC_Data.zip
DRYAD_TOKEN_URL = "https://datadryad.org/oauth/token"
DRYAD_DOWNLOAD_URL = f"https://datadryad.org/api/v2/files/{DRYAD_FILE_ID}/download"

FISH_SPECIES = [
    "Chromis_chromis", "Diplodus_vulgaris", "Coris_julis", "Symphodus_tinca", "Spicara_smaris",
    "Diplodus_sargus", "Symphodus_roissali", "Mugilidae", "Antherina", "Seriola", "Boops_boops",
    "Thalassoma_pavo", "Oblada_melanura", "Labrus_merula", "Labrus_mixtus", "Mullus_surmuletus",
    "Sarpa_salpa", "Labrus_viridis", "Apogon_imberbis", "Epinephelus_marginatus",
    "Muraena_helena", "Gobidae", "Serranus_scriba", "Scorpoeana_scrofa", "Anthias_anthias",
    "Symphodus_rostratus", "Serranus_cabrilla", "Sciaena_umbra", "Sparus aurata", "Trachurus sp.",
    "Dicentrachus labrax", "Symphodus_mediterraneus", "Sphyraena sphyraena",
    "Spondyliosoma_cantharus", "Blennidae", "Symphodus_melanocercus", "Spicara_maena",
    "Phycis_phycis", "Conger_conger", "Diplodus_anularis",
]
# Unlisted classes are discarded: background (hard negatives), diver, drone, and "fish"
# (its 4 boxes are seagrass leaves)
CATEGORIES_FILTER = {
    **{species: "fish" for species in FISH_SPECIES},
    "Other": "fish",
    "jellyfish": "non-fish",
    "seaurchin": "non-fish",
}
SOURCE_SPLITS = ["testSet", "trainSet"]  # testSet first: its copy of duplicated frames is kept
TRAIN_SOURCE_SPLIT = "trainSet"

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_download_path / "autoUC_Data.zip"
data_t_path = raw_download_path / "autoUC_Data" / "Data_T"
extraction_done_path = data_t_path / ".extraction_complete"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "med_fish_census_mapped_annotations.json"


def read_dryad_credentials() -> dict:
    """DRYAD_* variables from the environment, falling back to the repo's .env file"""
    credentials = {}
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().startswith("DRYAD_"):
                credentials[key.strip()] = value.strip().strip("\"'")
    credentials.update({k: v for k, v in os.environ.items() if k.startswith("DRYAD_")})
    return credentials


def get_dryad_download_request(credentials: dict):
    """
    Returns a function that gives (url, headers) for the zip: a fresh bearer token is used to
    resolve the API download link to its signed S3 URL, which is called again when it expires.
    """

    def get_token() -> str:
        if "DRYAD_CLIENT_ID" in credentials and "DRYAD_CLIENT_SECRET" in credentials:
            response = requests.post(DRYAD_TOKEN_URL, data={
                "client_id": credentials["DRYAD_CLIENT_ID"],
                "client_secret": credentials["DRYAD_CLIENT_SECRET"],
                "grant_type": "client_credentials",
            }, timeout=60)
            response.raise_for_status()
            return response.json()["access_token"]
        return credentials["DRYAD_TOKEN"]

    def get_request():
        headers = {"Authorization": f"Bearer {get_token()}"}
        response = requests.get(DRYAD_DOWNLOAD_URL, headers=headers, allow_redirects=False, timeout=60)
        if response.is_redirect:
            return response.headers["Location"], {}
        response.raise_for_status()
        return DRYAD_DOWNLOAD_URL, headers

    return get_request


def download_data():
    """Extract Data_T from the local zip if there is one, otherwise from the remote zip"""
    if extraction_done_path.exists():
        print(f"Data already extracted: {data_t_path}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    credentials = read_dryad_credentials()

    if zip_path.exists():
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(
                f"{zip_path} exists but is not a complete zip. If a manual download is still "
                "running, wait for it to finish; otherwise delete the file and rerun."
            )
        print(f"Extracting Data_T from the local zip {zip_path}")
        zip_file = open(zip_path, "rb")
    elif "DRYAD_TOKEN" in credentials or (
        "DRYAD_CLIENT_ID" in credentials and "DRYAD_CLIENT_SECRET" in credentials
    ):
        print(f"Extracting Data_T from the remote zip (Dryad file {DRYAD_FILE_ID})")
        zip_file = HttpRangeFile(get_dryad_download_request(credentials))
    else:
        raise RuntimeError(f"""
Dryad only serves files to signed-in browsers or to API clients with a token. Either:

  1. Use the Dryad API (only the 4.4 GB Data_T folder is downloaded, not the 121.7 GB zip):
     - Sign in to https://datadryad.org with ORCID, go to My account > API and create an
       API account
     - Put its credentials in the repo's .env file (git-ignored) or in the environment:
         DRYAD_CLIENT_ID=...
         DRYAD_CLIENT_SECRET=...
       (or DRYAD_TOKEN=<a token copied from the account page>, valid for 10 hours)

  2. Or download the whole autoUC_Data.zip (121.7 GB) manually from
       {DATA_PAGE_URL}
     and move it to {zip_path}
     The script picks it up automatically and extracts Data_T from it.

Then rerun this script.""")

    with zip_file, zipfile.ZipFile(zip_file) as zf:
        # Data_T plus the class name lists, which are at the zip root
        members = sorted(
            (
                m for m in zf.infolist()
                if not m.is_dir() and (
                    m.filename.startswith("autoUC_Data/Data_T/")
                    or re.fullmatch(r"autoUC_Data/labels_\w+\.txt", m.filename)
                )
            ),
            key=lambda m: m.header_offset,  # archive order: one Range request per block
        )
        for member in tqdm(members, desc="Extracting Data_T"):
            zf.extract(member, raw_download_path)

    extraction_done_path.touch()
    print(f"Extracted {len(members)} files to {data_t_path}")


def read_class_names(file_name: str) -> list[str]:
    with open(data_t_path.parent / file_name, encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


def get_recording_id(video: str) -> str:
    """GoPro recording of a clip: "GH010003_2" -> "GH010003", "C0005" -> "C0005" """
    return re.match(r"[A-Z]+\d+", video).group(0)


def build_coco():
    """
    COCO file over the unique Data_T frames, with boxes from labels_allSpecies (or
    labels_Reduced when there is no labels_allSpecies file). Each image records the authors'
    split in "source_split" and its video in "video".
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    label_sets = {
        "labels_allSpecies": read_class_names("labels_allSpecies.txt"),
        "labels_Reduced": read_class_names("labels_Reduced.txt"),
    }
    category_names = list(dict.fromkeys(label_sets["labels_allSpecies"] + label_sets["labels_Reduced"]))
    category_ids = {name: i + 1 for i, name in enumerate(category_names)}

    images, annotations = [], []
    duplicates = 0
    fallback_label_images = 0
    seen = set()
    for source_split in SOURCE_SPLITS:
        image_paths = sorted((data_t_path / source_split / "images").glob("*.jpg"))
        for image_path in tqdm(image_paths, desc=source_split):
            if image_path.name in seen:
                duplicates += 1
                continue
            seen.add(image_path.name)

            for label_set, class_names in label_sets.items():
                label_path = data_t_path / source_split / label_set / f"{image_path.stem}.txt"
                if label_path.exists():
                    break
            else:
                raise FileNotFoundError(f"No labels for {image_path}")
            fallback_label_images += label_set == "labels_Reduced"

            shutil.copy2(image_path, coco_images_path / image_path.name)
            with Image.open(image_path) as im:
                width, height = im.size
            image_id = len(images) + 1
            video = image_path.stem.rsplit("_", 1)[0]
            images.append({
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "source_split": source_split,
                "video": video,
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
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_ids[class_names[int(class_id)]],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                })

    with open(coco_annotations_path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": name} for name, i in category_ids.items()],
        }, f, indent=2)

    print(f"Skipped {duplicates} frames that are in both of the authors' splits (kept once)")
    print(f"Used labels_Reduced for {fallback_label_images} images without labels_allSpecies")
    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


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
    # Authors' split by video (testSet -> val), with recordings that have clips in both of
    # their splits moved entirely to val
    with open(mapped_annotations_path) as f:
        images = json.load(f)["images"]

    recordings_by_split = {
        split: {get_recording_id(image["video"]) for image in images if image["source_split"] == split}
        for split in SOURCE_SPLITS
    }
    shared_recordings = set.intersection(*recordings_by_split.values())
    print(f"Recordings in both of the authors' splits, moved to val: {sorted(shared_recordings)}")

    train_images = {
        image["file_name"]
        for image in images
        if image["source_split"] == TRAIN_SOURCE_SPLIT
        and get_recording_id(image["video"]) not in shared_recordings
    }

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
