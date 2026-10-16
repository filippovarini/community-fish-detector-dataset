"""
Newfoundland Marine Refuge Fish Classification Dataset (N-MARINE)
Source: https://open.canada.ca/data/en/dataset/2ae46860-f82a-4127-bb1f-b02e36ef6a70
Split logic: By camera deployment. Videos are named <camera>_<YYYYMMDD>_<HHMMSS> and
    consecutive 5-minute videos of one deployment show the same baited crab pot, so whole
    deployments go to one split. The 5 deployments are camera 097 on 16-17 Jul, 097 on 19 Jul,
    116 on 21 Jul, 130 on 19 Jul and 136 on 19 Jul; camera 097 on 19 Jul (~17% of the
    annotated frames) is the validation set, the rest is train.
Category mapping:
  fish: Cod, Roughhead_grenadier, Atlantic_halibut, Redfish_mentella, Thorny_skate,
        Striped_wolffish, Spinytail_skate, Wolffish (northern wolffish)
  non-fish: Whelk (Buccinum undatum, a crawling sea snail ~10 cm)
  discard: none
The 3,000 background frames (no annotated organisms) are kept as images without annotations.
"""

import json
import struct
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    add_dataset_shortname_prefix_to_image_names,
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "n_marine"
DATA_URL = (
    "https://api-proxy.edh-cde.dfo-mpo.gc.ca/catalogue/records/"
    "2ae46860-f82a-4127-bb1f-b02e36ef6a70/attachments"
)
ANNOTATIONS_ZIP = "nmarine_csv.zip"
IMAGE_ZIPS = [
    "2021-07-16_P1.zip",
    "2021-07-16_P2.zip",
    "2021-07-17.zip",
    *[f"2021-07-19_P{i}.zip" for i in range(1, 9)],
    *[f"2021-07-21_P{i}.zip" for i in range(1, 4)],
]
CATEGORIES_FILTER = {
    "Cod": "fish",  # Gadus morhua
    "Roughhead_grenadier": "fish",  # Macrourus berglax
    "Atlantic_halibut": "fish",  # Hippoglossus hippoglossus
    "Redfish_mentella": "fish",  # Sebastes mentella
    "Thorny_skate": "fish",  # Amblyraja radiata
    "Striped_wolffish": "fish",  # Anarhichas lupus
    "Spinytail_skate": "fish",  # Bathyraja spinicauda
    "Wolffish": "fish",  # Anarhichas denticulatus (northern wolffish)
    "Whelk": "non-fish",  # Buccinum undatum
}
# (camera, date) deployments in the validation set; camera 097's 16 Jul deployment runs past
# midnight, so its 17 Jul videos belong to the same (train) deployment
VAL_DEPLOYMENTS = {("097", "20210719")}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
csv_dir = raw_download_path / "nmarine_csv"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "n_marine_mapped_annotations.json"


def download_data():
    raw_download_path.mkdir(parents=True, exist_ok=True)
    for name in [ANNOTATIONS_ZIP, *IMAGE_ZIPS]:
        download_file_resumable(f"{DATA_URL}/{name}", raw_download_path / name)

    if not csv_dir.exists():
        with zipfile.ZipFile(raw_download_path / ANNOTATIONS_ZIP) as z:
            z.extractall(raw_download_path)


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from the PNG IHDR chunk."""
    return struct.unpack(">II", data[16:24])


def csv_to_coco():
    """
    Build one COCO dataset from the per-video CSVs. Images are read straight from the zips
    (<date>/<video>/frameNNNNNN.png) and saved as <video>_frameNNNNNN.png; frames with no
    CSV row are the background images and get no annotations.
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    df = pd.concat(
        [pd.read_csv(f).assign(video=f.stem) for f in sorted(csv_dir.glob("*.csv"))],
        ignore_index=True,
    )
    df["file_name"] = df["video"] + "_" + df["img"]
    category_names = sorted(df["marine_class"].unique())
    category_ids = {name: i for i, name in enumerate(category_names, start=1)}

    images = {}
    for zip_name in IMAGE_ZIPS:
        with zipfile.ZipFile(raw_download_path / zip_name) as z:
            members = [m for m in z.namelist() if m.endswith(".png")]
            for member in tqdm(members, desc=zip_name):
                video, frame = member.split("/")[-2:]
                file_name = f"{video}_{frame}"
                assert file_name not in images, f"Duplicate image {file_name}"
                image_path = coco_images_path / file_name
                data = z.read(member)
                if not image_path.exists():
                    image_path.write_bytes(data)
                width, height = png_size(data)
                images[file_name] = {
                    "id": len(images) + 1,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }

    missing = set(df["file_name"]) - set(images)
    assert not missing, f"{len(missing)} annotated images not in the zips, e.g. {sorted(missing)[:5]}"

    annotations = []
    for row in df.itertuples():
        image = images[row.file_name]
        # Some boxes extend a few pixels past the frame
        x1, y1 = max(row.TL_x, 0), max(row.TL_y, 0)
        x2, y2 = min(row.BR_x, image["width"]), min(row.BR_y, image["height"])
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image["id"],
            "category_id": category_ids[row.marine_class],
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
        })

    coco = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for name, i in category_ids.items()],
    }
    with open(coco_annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    annotated = len({a["image_id"] for a in annotations})
    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)} ({annotated} annotated, {len(images) - annotated} background)")
    print(f"Number of annotations: {len(annotations)} (of {len(df)} CSV rows)")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    csv_to_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def get_deployment(image_name: str) -> tuple[str, str]:
    """n_marine_097_20210717_012344_frame000123.png -> ("097", "20210716")"""
    camera, date = Path(image_name).name.removeprefix(f"{DATASET_SHORTNAME}_").split("_")[:2]
    if (camera, date) == ("097", "20210717"):
        date = "20210716"
    return camera, date


def dataset_splitting():
    should_the_image_be_included_in_train_set = (
        lambda image_path: get_deployment(image_path) not in VAL_DEPLOYMENTS
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
