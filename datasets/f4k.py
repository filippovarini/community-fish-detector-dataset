"""
F4K (Fish4Knowledge) Detection/Tracking Dataset
Source: https://studentiunict-my.sharepoint.com/...
Split logic: By video ID (106-109 val, 110-124 train)
Categories kept: fish

Manual download required: download the zip file from the source URL
and place it in fish-datasets/data/raw/f4k/

NOTE: This script requires ffmpeg for video frame extraction.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    extract_downloaded_file,
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    convert_coco_annotations_from_0_indexed_to_1_indexed,
    copy_images_to_processing,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "f4k"
CATEGORIES_FILTER = ["fish"]

settings = Settings()

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = processing_dir / "annotations.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME / "f4k_detection_tracking"
input_images_dir = settings.raw_dir / DATASET_SHORTNAME / "coco"


def find_all_videos(path):
    """Find all video files in the given path."""
    videos = []
    for filename in os.listdir(path):
        if filename.endswith(".mp4"):
            videos.append(filename.removesuffix(".mp4"))
    return videos


def get_all_categories():
    """Return the category mapping for F4K."""
    return {category: id for id, category in enumerate((
        'fish', 'open_sea', 'sea', 'rocks', 'coral', 'plant', 'dark_area', 'other', 'algae'
    ))}


def extract_keyframes(label_path, video_path, video_name, output_dir, category_id_map, annotation_id):
    """Extract keyframes from a video based on XML annotations and create COCO format."""
    tree = ET.parse(label_path)
    root = tree.getroot()

    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    cap = cv2.VideoCapture(str(video_path))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    correction = 3 if video_name.endswith("122") else 0

    for frame in root.findall("frame"):
        frame_id = int(frame.attrib["id"]) + correction
        image_id = f"video_gt_{video_name}_frame_{frame_id}"
        file_name = image_id + '.jpg'
        save_path = os.path.join(str(output_dir), file_name)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame_image = cap.read()

        if not ret:
            print(f"Failed to read frame {frame_id} from {video_path}")
            continue

        cv2.imwrite(save_path, frame_image)

        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": vid_width,
            "height": vid_height,
        })

        for obj in frame.findall("object"):
            category_name = obj.attrib["objectType"]
            if category_name in category_id_map:
                category_id = category_id_map[category_name]
            else:
                continue

            contour = obj.find("contour").text.strip().split(',')
            segmentation = []
            for point in contour:
                x, y = map(float, point.split())
                segmentation.extend([x, y])

            xs = segmentation[::2]
            ys = segmentation[1::2]
            xmin = min(xs)
            ymin = min(ys)
            width = max(xs) - xmin
            height = max(ys) - ymin

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1

    cap.release()
    return coco, annotation_id


def extract_data():
    """Extract the zip file and process videos into frames with COCO annotations."""
    zip_path = settings.raw_dir / DATASET_SHORTNAME / "f4k_detection_tracking.zip"
    if zip_path.exists():
        extract_downloaded_file(zip_path, settings.raw_dir / DATASET_SHORTNAME)

    output_dir = input_images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = raw_data_dir / "videos"
    labels_dir = raw_data_dir / "labels"
    category_id_map = get_all_categories()

    all_coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cat_id, "name": cat_name} for cat_name, cat_id in category_id_map.items()],
    }
    annotation_id = 1

    video_names = find_all_videos(str(videos_dir))
    for video_name in tqdm(video_names, desc="Processing videos"):
        video_path = videos_dir / f"{video_name}.mp4"
        label_path = labels_dir / f"{video_name}.xml"

        if not label_path.exists():
            print(f"Label file not found: {label_path}")
            continue

        coco, annotation_id = extract_keyframes(
            str(label_path), str(video_path), video_name, output_dir,
            category_id_map, annotation_id,
        )
        all_coco["images"].extend(coco["images"])
        all_coco["annotations"].extend(coco["annotations"])

    coco_output_path = output_dir / "annotations_coco.json"
    with open(coco_output_path, "w") as f:
        json.dump(all_coco, f, indent=2)

    print(f"Saved {len(all_coco['images'])} images and {len(all_coco['annotations'])} annotations")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    if not input_images_dir.exists():
        extract_data()

    images_path.mkdir(parents=True, exist_ok=True)
    copy_images_to_processing(DATASET_SHORTNAME, input_images_dir)

    shutil.move(str(images_path / "annotations_coco.json"), str(annotations_path))

    corrected_annotations_path = processing_dir / "corrected_annotations.json"
    convert_coco_annotations_from_0_indexed_to_1_indexed(annotations_path, corrected_annotations_path)

    compress_annotations_to_single_category(
        corrected_annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path, compressed_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # val = videos 106-109, train = videos 110-124
    should_the_image_be_included_in_train_set = (
        lambda image_path: int(Path(image_path).stem.split("_")[3]) > 109
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
        images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


def main():
    # 1. DOWNLOAD + EXTRACT
    if not raw_data_dir.exists():
        extract_data()

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
