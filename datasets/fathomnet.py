"""
FathomNet Dataset
Source: FathomNet API (fathomnet-generate CLI)
Split logic: Random (no location/camera grouping available)
Categories kept: All (filtered to fish concepts at download time)

Note: Download takes ~12 hours. Uses fathomnet API to find all fish concepts
(Actinopterygii, Sarcopterygii, Chondrichthyes, Myxini descendants) and
downloads bounding box annotations via fathomnet-generate.
"""

import random
import subprocess
from pathlib import Path

from fathomnet.api import boundingboxes, worms

from datasets.settings import Settings
from datasets.utils import (
    compress_annotations_to_single_category,
    add_dataset_shortname_prefix_to_image_names,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "fathomnet"
CATEGORIES_FILTER = None

settings = Settings()


def download_data(data_dir: Path):
    print(f"Downloading data to {data_dir}")
    annotations_path = data_dir / "dataset.json"
    images_path = data_dir / "images"
    print(f"Annotations will be saved to {annotations_path}")
    print(f"Images will be downloaded to {images_path}")

    if data_dir.exists() and len(list(data_dir.glob("*.json"))) > 0:
        print(f"Dataset already exists in {data_dir}")
        return annotations_path, images_path

    # Choose the root concepts for fish
    root_concepts = ["Actinopterygii", "Sarcopterygii", "Chondrichthyes", "Myxini"]
    print(f"Root concepts: {', '.join(root_concepts)}\n")

    # Get all descendants of the root concepts
    fish_concepts = set(root_concepts)
    for rc in root_concepts:
        descendant_concepts = worms.get_descendants_names(rc)
        fish_concepts.update(descendant_concepts)
        print(f"Added {len(descendant_concepts)} descendants of {rc}")

    # Find annotated concepts in FathomNet
    fathomnet_concepts = set(boundingboxes.find_concepts())

    # Compute the intersection of fish concepts and FathomNet concepts
    fish_concepts_in_fathomnet = fish_concepts & fathomnet_concepts
    print(f"\nFound {len(fish_concepts_in_fathomnet)} fish concepts in FathomNet")

    # Write the fish concepts to a file
    fish_concepts_file = data_dir / "concepts.txt"
    with fish_concepts_file.open("w") as f:
        f.write("\n".join(sorted(fish_concepts_in_fathomnet)))
    print(f"Wrote selected concepts to {fish_concepts_file}")

    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True, parents=True)
    images_path.mkdir(exist_ok=True, parents=True)

    cmd = [
        "fathomnet-generate",
        "-v",
        "--format",
        "coco",
        "--concepts-file",
        str(fish_concepts_file),
        "--output",
        str(data_dir),
        "--img-download",
        str(images_path),
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return annotations_path, images_path


def get_list_of_images_to_include_in_train_set(image_folder: Path) -> list[str]:
    all_images = list(image_folder.glob("*.png"))
    train_ratio = 1 - settings.train_val_split_ratio
    train_size = int(len(all_images) * train_ratio)
    train_images = random.sample(all_images, train_size)
    return [image.name for image in train_images]


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    annotations_path, raw_images_path = download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    compressed_annotations_path = processing_dir / "compressed_annotations.json"
    compressed_annotations_path = compress_annotations_to_single_category(
        annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=raw_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(raw_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
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
