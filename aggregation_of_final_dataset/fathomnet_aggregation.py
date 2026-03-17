"""
Note: We use a different naming convention for fathomnet than other packages
(i.e. fathomnet_aggregation.py instead of fathomnet.py) because fathomnet is a
package that is not part of the fish-datasets repository and would cause a name
conflict if we used fathomnet.py.
"""
from pathlib import Path
import random

from data_preview.visualize_fathomnet import download_data, DATASET_SHORTNAME
from aggregation_of_final_dataset.utils import (
    compress_annotations_to_single_category,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
)
from settings import Settings

settings = Settings()


def get_list_of_images_to_include_in_train_set(image_folder: Path) -> list[str]:
    all_images = list(image_folder.glob("*.png"))
    train_ratio = 1 - settings.train_val_split_ratio
    train_size = int(len(all_images) * train_ratio)
    train_images = random.sample(all_images, train_size)
    train_images = [image.name for image in train_images]
    return train_images

def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    annotations_path, raw_images_path = download_data(raw_download_path)

    # # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    categories_to_keep = None
    compressed_annotations_path = processing_dir / "compressed_annotations.json"
    compressed_annotations_path = compress_annotations_to_single_category(
        annotations_path, categories_to_keep, compressed_annotations_path
    )

    # 3. FINAL
    add_dataset_shortname_prefix_to_image_names(
        images_path=raw_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )
    
    # Build Logic to split into train and val based on camera name
    train_camera_names = get_list_of_images_to_include_in_train_set(raw_images_path)
    should_the_image_be_included_in_train_set = (
        lambda image_name: image_name in train_camera_names
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
