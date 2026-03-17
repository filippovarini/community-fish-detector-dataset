from typing import Set
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_preview.visualize_noaa_puget import download_data, DATASET_SHORTNAME
from aggregation_of_final_dataset.utils import (
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    convert_coco_annotations_from_0_indexed_to_1_indexed,
    remove_dataset_shortname_prefix_from_image_filename,
)
from settings import Settings

settings = Settings()


def get_unique_camera_names(image_folder: Path) -> Set:
    camera_names = set()
    for image_path in image_folder.glob("*.jpg"):
        camera_name = remove_dataset_shortname_prefix_from_image_filename(
            image_path.stem, DATASET_SHORTNAME
        ).split("_")[2]
        camera_names.add(camera_name)

    return camera_names


def get_list_of_cameras_to_include_in_train_set(image_folder: Path) -> list[str]:
    # Split the camera names into train and val
    camera_names = list(get_unique_camera_names(image_folder))
    train_camera_names, _ = train_test_split(
        list(camera_names),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_camera_names


def main():
    # 1. RAW
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_images_path = raw_download_path / settings.images_folder_name
    raw_annotations_path = raw_download_path / "noaa_estuary_fish-2023.08.19.json"
    
    if not raw_images_path.exists() or not raw_annotations_path.exists():
        # Download NOAA Data in Raw Directory
        raw_download_path.mkdir(parents=True, exist_ok=True)
        download_data(raw_download_path)

    # # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    # Create COCO Dataset and store in intermediate directory
    # We compress all annotations into a single category: Fish
    annotations_path_1_indexed = (
        processing_dir / "noaa_puget_annotations_1_indexed.json"
    )
    convert_coco_annotations_from_0_indexed_to_1_indexed(
        raw_annotations_path, annotations_path_1_indexed
    )

    categories_to_keep = ["fish"]
    compressed_annotations_path = (
        processing_dir / "noaa_puget_compressed_annotations.json"
    )
    compressed_annotations_path = compress_annotations_to_single_category(
        annotations_path_1_indexed, categories_to_keep, compressed_annotations_path
    )

    # 3. FINAL
    add_dataset_shortname_prefix_to_image_names(
        images_path=raw_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # Build Logic to split into train and val based on camera name
    train_camera_names = get_list_of_cameras_to_include_in_train_set(raw_images_path)
    print(f"Train camera names: {train_camera_names}")
    should_the_image_be_included_in_train_set = (
        lambda image_name: remove_dataset_shortname_prefix_from_image_filename(
            image_name, DATASET_SHORTNAME
        ).split("_")[2]
        in train_camera_names
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
        raw_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
