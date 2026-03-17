from pathlib import Path
import json
import shutil

import tqdm
from sklearn.model_selection import train_test_split

from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    compress_annotations_to_single_category,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
)
from data_preview.visualize_mit_river_herring import download_data, DATASET_SHORTNAME


settings = Settings()


def get_list_of_deployments_to_include_in_train_set(images_path: Path) -> list[str]:
    """
    Extract location and video names from the image filenames.
    """
    all_deployments = list(
        set(
            [
                "_".join(
                    remove_dataset_shortname_prefix_from_image_filename(
                        img_path.name, DATASET_SHORTNAME
                    ).split("_")[:2]
                )
                for img_path in images_path.glob("*.PNG")
            ]
        )
    )
    train_deployments, _ = train_test_split(
        all_deployments,
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_deployments


def aggregate_images_in_one_folder(
    annotations_path: Path,
    images_path: Path,
    aggregated_annotations_path: Path,
    aggregated_images_path: Path,
):
    """
    The dataset is a collection of multiple coco datasets, one for each
    location and video.
    We join all into one single coco dataset.
    """
    if aggregated_images_path.exists():
        print(f"Aggregated images already exist: {aggregated_images_path}")
        return

    if aggregated_annotations_path.exists():
        print(f"Aggregated annotations already exist: {aggregated_annotations_path}")
        return

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    aggregated_images_path.mkdir(parents=True, exist_ok=True)

    for image in tqdm.tqdm(annotations["images"], total=len(annotations["images"])):
        original_image_path = images_path / image["file_name"]
        new_image_file_name = "_".join(Path(image["file_name"]).parts)
        new_image_path = aggregated_images_path / new_image_file_name
        shutil.copy(original_image_path, new_image_path)
        image["file_name"] = new_image_file_name

    with open(aggregated_annotations_path, "w") as f:
        json.dump(annotations, f)


def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    # raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    # raw_download_path.mkdir(parents=True, exist_ok=True)
    # annotations_path, images_path = download_data(raw_download_path)
    annotations_path, images_path = Path(
        "/mnt/data/dev/fish-datasets/data/raw/mit_river_herring/mit_river_herring/mit_sea_grant_river_herring.json"
    ), Path("/mnt/data/dev/fish-datasets/data/raw/mit_river_herring/mit_river_herring")

    # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    aggregated_annotations_path = processing_dir / settings.coco_file_name
    aggregated_images_path = processing_dir / settings.images_folder_name
    aggregate_images_in_one_folder(
        annotations_path,
        images_path,
        aggregated_annotations_path,
        aggregated_images_path,
    )

    categories_to_keep = None
    compressed_annotations_path = processing_dir / "compressed_annotations.json"
    compressed_annotations_path = compress_annotations_to_single_category(
        aggregated_annotations_path, categories_to_keep, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=aggregated_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. FINAL
    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    # Build Logic to split into train and val based on camera name
    train_camera_names = get_list_of_deployments_to_include_in_train_set(
        aggregated_images_path
    )
    should_the_image_be_included_in_train_set = (
        lambda image_name: "_".join(
            remove_dataset_shortname_prefix_from_image_filename(
                image_name, DATASET_SHORTNAME
            ).split("_")[:2]
        )
        in train_camera_names
    )

    split_coco_dataset_into_train_validation(
        aggregated_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
