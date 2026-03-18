"""
FishCLEF 2015 Dataset
Source: https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1
Split logic: By video ID (train_test_split on unique video filenames)
Categories kept: All (majority are "Null" = general fish)

Frame extraction from .flv videos. One frame per annotated frame in the XML.
"""

import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "fishclef"
DATA_URL = "https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1"
CATEGORIES_FILTER = None

settings = Settings()


def download_data(data_dir: Path):
    """Download and extract the fishclef dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract(data_dir, DATA_URL, DATASET_SHORTNAME)


def get_width_and_heigth_of_video(videos_dir: Path, video_id: str):
    """Get the width and height of a video."""
    video_path = videos_dir / f"{video_id}.flv"
    cap = cv2.VideoCapture(str(video_path))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return width, height


def convert_xml_to_coco(videos_dir: Path, xml_file, output_dir=None):
    """Converts a single XML annotation file to COCO JSON format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    video_id = Path(xml_file).stem
    width, height = get_width_and_heigth_of_video(videos_dir, video_id)

    images = []
    annotations = []
    categories = {}
    ann_id = 1

    for frame in root.findall("frame"):
        frame_id = frame.get("id")
        image_info = {
            "id": int(frame_id),
            "file_name": f"{video_id}_frame_{frame_id}.jpg",
            "width": width,
            "height": height,
        }
        images.append(image_info)

        for obj in frame.findall("object"):
            species = obj.get("fish_species")
            if species not in categories:
                categories[species] = len(categories) + 1

            x = int(obj.get("x"))
            y = int(obj.get("y"))
            w = int(obj.get("w"))
            h = int(obj.get("h"))

            ann = {
                "id": ann_id,
                "image_id": int(frame_id),
                "category_id": categories[species],
                "bbox": [x, y, w, h],
            }
            annotations.append(ann)
            ann_id += 1

    categories_list = [
        {"id": cat_id, "name": species}
        for species, cat_id in categories.items()
    ]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }

    if output_dir is None:
        output_dir = os.path.dirname(xml_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    output_json = os.path.join(output_dir, base_name + ".json")
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)

    return output_json


def convert_annotations(download_dir: Path, output_dir: Path):
    """Convert XML annotations to COCO format for both training and test sets."""
    training_annotations_path = download_dir / "fishclef_2015_release" / "training_set" / "gt"
    training_annotations_coco_path = output_dir / "fishclef_2015_release" / "training_set" / "gt_coco"
    training_videos_path = download_dir / "fishclef_2015_release" / "training_set" / "videos"

    test_annotations_path = download_dir / "fishclef_2015_release" / "test_set" / "gt"
    test_annotations_coco_path = output_dir / "fishclef_2015_release" / "test_set" / "gt_coco"
    test_videos_path = download_dir / "fishclef_2015_release" / "test_set" / "videos"

    if training_annotations_coco_path.exists() and test_annotations_coco_path.exists():
        print("Annotations already converted, skipping")
        return

    training_annotations_coco_path.mkdir(exist_ok=True, parents=True)
    test_annotations_coco_path.mkdir(exist_ok=True, parents=True)

    xml_files = training_annotations_path.rglob("*.xml")
    for xml_file in xml_files:
        output_json = convert_xml_to_coco(training_videos_path, xml_file, training_annotations_coco_path)
        print(f"Converted '{xml_file}' to '{output_json}'.")

    xml_files = test_annotations_path.rglob("*.xml")
    for xml_file in xml_files:
        output_json = convert_xml_to_coco(test_videos_path, xml_file, test_annotations_coco_path)
        print(f"Converted '{xml_file}' to '{output_json}'.")


def extract_frame(cap, frame_index):
    """Extract a specific frame from a video."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Frame {frame_index} could not be read")
    return frame


def merge_coco_datasets_into_single_dataset(annotations_paths: List[Path], output_path: Path):
    """Merges a list of COCO datasets into a single COCO dataset."""
    if not annotations_paths:
        raise ValueError("No annotation paths provided")

    if output_path.exists():
        print(f"Output file already exists at {output_path}")
        return output_path

    image_id_counter = 1
    annotation_id_counter = 1
    category_id_counter = 1
    category_names_to_id = {}

    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    for path in annotations_paths:
        print(f"Merging {path}")
        if not path.exists():
            print(f"Warning: File {path} does not exist, skipping")
            continue

        with open(path, "r") as f:
            try:
                coco_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: File {path} is not valid JSON, skipping")
                continue

        old_image_id_to_new_image_id = {}
        old_category_id_to_new_category_id = {}

        for image in coco_data.get("images", []):
            old_image_id = image["id"]
            old_image_id_to_new_image_id[old_image_id] = image_id_counter
            image["id"] = image_id_counter
            merged_coco["images"].append(image)
            image_id_counter += 1

        unique_categories = []
        for category in coco_data.get("categories", []):
            old_category_id = category["id"]
            category_name = category["name"]

            if category_name in category_names_to_id:
                old_category_id_to_new_category_id[old_category_id] = category_names_to_id[category_name]
            else:
                category_names_to_id[category_name] = category_id_counter
                old_category_id_to_new_category_id[old_category_id] = category_id_counter
                category["id"] = category_id_counter
                unique_categories.append(category)
                category_id_counter += 1

        merged_coco["categories"].extend(unique_categories)

        for annotation in coco_data.get("annotations", []):
            try:
                old_image_id = annotation["image_id"]
                old_category_id = annotation["category_id"]
                annotation["image_id"] = old_image_id_to_new_image_id[old_image_id]
                annotation["category_id"] = old_category_id_to_new_category_id[old_category_id]
                annotation["id"] = annotation_id_counter
                annotation["iscrowd"] = 0
                _, _, w, h = annotation["bbox"]
                annotation["area"] = w * h
                merged_coco["annotations"].append(annotation)
                annotation_id_counter += 1
            except KeyError as e:
                print(f"Warning: Invalid annotation (missing {e}), skipping")

    print(f"Merged dataset contains:")
    print(f"  - {len(merged_coco['images'])} images")
    print(f"  - {len(merged_coco['annotations'])} annotations")
    print(f"  - {len(merged_coco['categories'])} categories")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged_coco, f, indent=2)

    return output_path


def extract_frames_from_videos(download_dir: Path, frames_dir: Path, coco_data: dict):
    """Extract frames from videos and save them to disk."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    if frames_dir.exists() and len(list(frames_dir.glob("*"))) > 0:
        print(f"Frames directory already exists at {frames_dir}")
        return frames_dir

    video_name_to_video_path = {video_path.stem: video_path for video_path in download_dir.rglob("*.flv")}
    print(f"Found {len(video_name_to_video_path)} videos")

    extracted_frames = set()

    for image_info in coco_data["images"]:
        frame_filename = image_info["file_name"]
        frame_id = int(Path(frame_filename).stem.split("_frame_")[1])
        frame_name = Path(frame_filename).stem.split("_frame_")[0]
        if frame_name not in video_name_to_video_path:
            print(f"Frame {frame_name} not found in {video_name_to_video_path}")
            continue

        video_path = video_name_to_video_path[frame_name]
        print(f"Extracting frame {frame_id} from {video_path}", end="\r")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        try:
            frame = extract_frame(cap, frame_id)
            output_path = frames_dir / frame_filename
            cv2.imwrite(str(output_path), frame)
            extracted_frames.add(str(output_path))
        except Exception as e:
            print(f"Error extracting frame {frame_id} from {video_path}: {e}")

        cap.release()

    print(f"\nExtracted {len(extracted_frames)} frames")
    return frames_dir


def get_list_of_videos_to_include_in_train_set(raw_download_path: Path):
    all_video_ids = [video_path.stem for video_path in raw_download_path.rglob("*.flv")]
    train_video_ids, _ = train_test_split(
        all_video_ids,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_video_ids


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    convert_annotations(raw_download_path, processing_dir)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    all_annotation_paths = list(processing_dir.rglob("*.json"))
    print(f"Merging {len(all_annotation_paths)} annotation files")
    merge_coco_datasets_into_single_dataset(all_annotation_paths, coco_annotations_path)

    with open(coco_annotations_path, "r") as f:
        coco_annotations = json.load(f)

    extract_frames_from_videos(raw_download_path, coco_images_path, coco_annotations)

    compressed_annotations_path = (
        processing_dir / "fishclef_compressed_annotations.json"
    )
    compressed_annotations_path = compress_annotations_to_single_category(
        coco_annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path, compressed_annotations_path, DATASET_SHORTNAME
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_videos_ids = get_list_of_videos_to_include_in_train_set(raw_download_path)
    should_the_image_be_included_in_train_set = (
        lambda image_name: remove_dataset_shortname_prefix_from_image_filename(
            image_name, DATASET_SHORTNAME
        ).split("_frame_")[0]
        in train_videos_ids
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
