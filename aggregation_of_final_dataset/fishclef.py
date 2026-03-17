import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_preview.visualize_fishclef import (
    DATASET_SHORTNAME,
    download_data,
    convert_annotations,
    extract_frames_from_videos,
    merge_coco_datasets_into_single_dataset,
)
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import compress_annotations_to_single_category, split_coco_dataset_into_train_validation


settings = Settings()


def get_list_of_videos_to_include_in_train_set(raw_download_path: Path):
    all_video_ids = [video_path.stem for video_path in raw_download_path.rglob("*.flv")]
    train_video_ids, _ = train_test_split(
        all_video_ids,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_video_ids

def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    # Convert xml to coco format
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
    
    # # Keep all categories. Majority of bboxes are of category "Null" which just represents "general fish"
    categories_to_keep = None
    compressed_annotations_path = (
        processing_dir / "fishclef_compressed_annotations.json"
    )
    compressed_annotations_path = compress_annotations_to_single_category(
        coco_annotations_path, categories_to_keep, compressed_annotations_path
    )
    
    # 3. FINAL
    # Build Logic to split into train and val based on camera name
    train_videos_ids = get_list_of_videos_to_include_in_train_set(raw_download_path)
    should_the_image_be_included_in_train_set = (
        lambda image_name: image_name.split("_frame_")[0] in train_videos_ids
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
