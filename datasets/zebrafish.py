"""
AAU Zebrafish ReID Dataset
Source: Kaggle - aalborguniversity/aau-zebrafish-reid
Split logic: By video (Vid1 -> train, Vid2 -> val, hardcoded)
Categories kept: All (only Zebrafish)

Not the ideal split ratio, but we give priority to not polluting
training/val with images from the same video.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import kagglehub
import pandas as pd
from PIL import Image
import tqdm

from datasets.settings import Settings
from datasets.utils import (
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "zebrafish"
DATASET_URL = "aalborguniversity/aau-zebrafish-reid"
CATEGORIES_FILTER = None

settings = Settings()


def download_data(download_path: Path):
    """Download the dataset directly from Kaggle."""
    if download_path.exists() and any(download_path.iterdir()):
        print("Data already downloaded in the directory:", download_path)
        return download_path

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(DATASET_URL)
    shutil.move(path, str(download_path))
    return download_path


def clean_annotations_and_get_df(data_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """Clean and process the annotations."""
    data_path = data_dir / "2" / "data"
    annotation_path = data_dir / "2" / "annotations.csv"

    data_df = pd.read_csv(annotation_path, sep=";")

    if len(list(data_path.iterdir())) == len(data_df):
        print("Number of images in the data directory and the dataframe are equal:", len(data_df))
    else:
        print("Number of images in the data directory and the dataframe are not equal")
        print("Number of images in the data directory:", len(list(data_path.iterdir())))
        print("Number of annotations in the dataframe:", len(data_df))

    combined_col = "Right,Turning,Occlusion,Glitch"
    for idx, col in enumerate(combined_col.split(",")):
        data_df[col] = data_df[combined_col].apply(lambda x: x.split(",")[idx])

    ws = data_df["Lower right corner X"] - data_df["Upper left corner X"]
    hs = data_df["Lower right corner Y"] - data_df["Upper left corner Y"]

    data_df["bbox"] = [
        [x, y, w, h]
        for x, y, w, h in list(
            zip(
                data_df["Upper left corner X"].values,
                data_df["Upper left corner Y"].values,
                ws,
                hs,
            )
        )
    ]

    data_df["path"] = data_path / data_df["Filename"]
    data_df["Object ID"] = data_df["Object ID"].astype(str)
    data_df["label"] = data_df["Annotation tag"]
    data_df["image_id"] = data_df["Filename"].apply(lambda x: x.split(".")[0])

    data_df = data_df[["image_id", "label", "bbox", "path"]]
    data_df = (
        data_df.groupby("image_id")
        .agg({"label": list, "bbox": list, "path": list})
        .reset_index()
    )

    return data_df, data_path


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj


def dataframe_to_coco(df, output_json_path: Path):
    """Convert the DataFrame to COCO format and save to JSON."""
    if output_json_path.exists():
        print(f"COCO format JSON already exists at {output_json_path}")
        return output_json_path

    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    unique_labels = sorted(set([label for sublist in df["label"] for label in sublist]))
    label_to_id = {label: i + 1 for i, label in enumerate(unique_labels)}

    for label, cat_id in label_to_id.items():
        coco_format["categories"].append(
            {"id": cat_id, "name": label, "supercategory": "none"}
        )

    annotation_id = 1
    for idx, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["path"][0]

        with Image.open(image_path) as img:
            width, height = img.size

        coco_format["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for label, bbox in zip(row["label"], row["bbox"]):
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_id[label],
                    "bbox": convert_to_serializable(bbox),
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO format JSON saved to {output_json_path}")
    return output_json_path


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(raw_download_path)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    data_df, raw_images_path = clean_annotations_and_get_df(raw_download_path)
    dataframe_to_coco(data_df, coco_annotations_path)

    images_generator = raw_images_path.glob("*.png")
    total_images = list(images_generator)
    for image_path in tqdm.tqdm(total_images, total=len(total_images)):
        shutil.copy2(image_path, coco_images_path / image_path.name)

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    compress_annotations_to_single_category(
        coco_annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    should_the_image_be_included_in_train_set = (
        lambda image_name: remove_dataset_shortname_prefix_from_image_filename(
            image_name, DATASET_SHORTNAME
        ).startswith("Vid1")
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
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
