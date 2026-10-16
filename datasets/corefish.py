"""
CoReFish (Caribbean Coral Reef Fish)
Source: https://warplab.github.io/corefish/ (corefish_dataset.zip on Google Drive)
Split logic: The authors' split, which is by survey trip (year): their train split -> train,
    their val + test splits -> val. Every transect video and every survey trip lies in exactly
    one split.
Category mapping:
  fish: fish (coco_fishonly.json has the single category "fish"; the species-level labels in
        coco_species.json cover only a subset of frames and are all fish species, so they are
        not used)
  non-fish: none (only fish are annotated)
Images: 14,580 diver-transect frames (162 transects x 90 frames) from 5 reef sites in the US
    Virgin Islands. Frames without boxes (~25%) are kept as background images. Transect
    Summer2016/JoelsShoal30mTransects061016/P6100007_0m_26s is annotated in the COCO file but
    its images are missing from the zip (the folder P6100005_0m_20s is there instead and its
    content does not match the boxes), so those 90 frames are dropped.
"""

import json
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "corefish"
DATA_PAGE_URL = "https://drive.google.com/file/d/1LENr6D8XW8ivusF2riS0v27Grwa1Uj2q/view"
DATA_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1LENr6D8XW8ivusF2riS0v27Grwa1Uj2q&export=download&confirm=t"
)
CATEGORIES_FILTER = {
    "fish": "fish",
}
ZIP_ROOT = "corefish_dataset"
ZIP_IMAGES_DIR = f"{ZIP_ROOT}/images/images"
SOURCE_SPLITS = ["train", "val", "test"]
TRAIN_SOURCE_SPLITS = {"train"}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_download_path / "corefish_dataset.zip"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "corefish_mapped_annotations.json"


def download_data():
    # The 40 GB zip is kept and read directly; images are extracted straight into processing
    if zip_path.exists():
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(
                f"{zip_path} exists but is not a complete zip. If a manual download is still "
                "running, wait for it to finish; otherwise delete the file and rerun."
            )
        print(f"Data already downloaded: {zip_path}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_file_resumable(DATA_URL, zip_path)

    if not zipfile.is_zipfile(zip_path):
        # Google Drive returns a small HTML page instead of the zip, usually "Quota exceeded"
        with open(zip_path, "rb") as f:
            page = f.read(5000).decode(errors="replace")
        zip_path.unlink()
        title = re.search(r"<title>(.*?)</title>", page)
        raise RuntimeError(f"""
Google Drive did not return the zip but a page titled "{title.group(1) if title else page[:200]}".

The file is public, but Google Drive limits how often it can be downloaded without being
signed in. This dataset is popular and large, so the anonymous quota is often used up for
up to 24 hours ("Quota exceeded"). Downloads made while signed in to Google use your own
quota and usually work. Either:

  1. Download it in a browser where you are signed in to Google:
       {DATA_PAGE_URL}
     and move the file to {zip_path}

  2. Or download it on this machine with your browser's signed-in session:
     - In Chrome, open DevTools (F12) > Network and start the download from the page above
     - Right-click the "download?id=..." request > Copy > Copy as cURL, then cancel the
       browser download
     - On this machine, run the copied command with this appended:
         -L -C - -o {zip_path}
       (-C - resumes an interrupted download.) The command contains your Google login
       cookies: don't share or commit it.

  3. Or rerun this script later (the anonymous quota resets within 24 hours).

Then rerun this script: it skips the download when the zip is already there.""")


def flatten_image_path(relative_path: str) -> str:
    """
    "Jan2021_FishSurveys/Joel's Shoal Fish Surveys 1.21.2021/P1010017_0m_11s/frame_015.png"
    -> "Jan2021_FishSurveys__Joel_s_Shoal_Fish_Surveys_1.21.2021__P1010017_0m_11s__frame_015.png"
    """
    return "__".join(
        re.sub(r"[^A-Za-z0-9.-]+", "_", part) for part in relative_path.split("/")
    )


def build_coco():
    """
    Extract the annotated frames from the zip and write a COCO file with flattened image
    names. Each image records the authors' split in "source_split".
    """
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        coco = json.loads(zf.read(f"{ZIP_ROOT}/coco_fishonly.json"))
        source_split = {}
        for split in SOURCE_SPLITS:
            for line in zf.read(f"{ZIP_ROOT}/{split}_fishonly_split.txt").decode().splitlines():
                if line.strip():
                    source_split[line.strip().removeprefix("./images/")] = split
        zip_names = set(zf.namelist())

        images = []
        dropped_image_ids = set()
        for image in tqdm(coco["images"]):
            relative_path = image["file_name"]
            zip_name = f"{ZIP_IMAGES_DIR}/{relative_path}"
            if zip_name not in zip_names:
                dropped_image_ids.add(image["id"])
                continue

            file_name = flatten_image_path(relative_path)
            target_path = coco_images_path / file_name
            if not target_path.exists():
                with zf.open(zip_name) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            with Image.open(target_path) as im:
                assert im.size == (image["width"], image["height"]), (relative_path, im.size)

            images.append({
                "id": image["id"],
                "file_name": file_name,
                "width": image["width"],
                "height": image["height"],
                "source_split": source_split[relative_path],
            })

    file_names = [image["file_name"] for image in images]
    assert len(file_names) == len(set(file_names)), "Flattened image names collide"

    dropped_folders = sorted({
        image["file_name"].rsplit("/", 1)[0]
        for image in coco["images"] if image["id"] in dropped_image_ids
    })
    print(f"Dropped {len(dropped_image_ids)} annotated images missing from the zip: {dropped_folders}")

    annotations = [
        {k: annotation[k] for k in ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]}
        for annotation in coco["annotations"]
        if annotation["image_id"] not in dropped_image_ids
    ]

    with open(coco_annotations_path, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": coco["categories"]},
            f,
            indent=2,
        )

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
    # Authors' split by survey trip: their train -> train, their val + test -> val
    with open(mapped_annotations_path) as f:
        train_images = {
            image["file_name"]
            for image in json.load(f)["images"]
            if image["source_split"] in TRAIN_SOURCE_SPLITS
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
