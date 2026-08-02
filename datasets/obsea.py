"""
OBSEA Dataset
Source: https://doi.org/10.1594/PANGAEA.946149
Split logic: By month; selected months held out for validation
Categories kept: All fish species from the PANGAEA annotation table
"""

import csv
import json
import os
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from datasets.settings import Settings
from datasets.utils.coco import compress_annotations_to_single_category
from datasets.utils.split import split_coco_dataset_into_train_validation
from datasets.utils.images import add_dataset_shortname_prefix_to_image_names


DATASET_SHORTNAME = "obsea"
PANGAEA_TABLE_NAME = "obsea_fish_2013_14.tab"
PANGAEA_IMAGE_ARCHIVE_NAME = "obsea_fish_2013_14_allfiles.zip"

# Hold out complete months to avoid using the random split from derived YOLO exports.
VALIDATION_MONTHS = {
    "2013-04",
    "2013-08",
    "2013-12",
    "2014-04",
    "2014-08",
    "2014-12",
}

settings = Settings()


def read_pangaea_table(table_path: Path) -> list[dict]:
    """
    Read the PANGAEA .tab file, skipping the leading metadata comment block.
    """
    if not table_path.exists():
        raise FileNotFoundError(f"OBSEA PANGAEA table not found at {table_path}")

    with open(table_path, "r", newline="") as f:
        for line in f:
            if line.startswith("Event\tDate/Time\tIMAGE\tSpecies"):
                fieldnames = line.rstrip("\n").split("\t")
                reader = csv.DictReader(f, fieldnames=fieldnames, delimiter="\t")
                return list(reader)

    raise ValueError(f"Could not find PANGAEA data header in {table_path}")


def extract_images_from_archive(
    image_archive_path: Path,
    image_filenames: list[str],
    output_images_path: Path,
) -> None:
    if not image_archive_path.exists():
        raise FileNotFoundError(f"OBSEA image archive not found at {image_archive_path}")

    output_images_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(image_archive_path) as image_archive:
        archive_name_by_filename = {
            Path(archive_name).name: archive_name
            for archive_name in image_archive.namelist()
            if not archive_name.endswith("/")
        }

        missing_images = [
            image_filename
            for image_filename in image_filenames
            if image_filename not in archive_name_by_filename
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} images from the annotation table were not "
                f"found in {image_archive_path}. First missing image: {missing_images[0]}"
            )

        for i, image_filename in enumerate(image_filenames, start=1):
            if i == 1 or i % 500 == 0 or i == len(image_filenames):
                print(f"Extracting image {i} of {len(image_filenames)}")
            output_image_path = output_images_path / image_filename
            with image_archive.open(archive_name_by_filename[image_filename]) as source:
                with open(output_image_path, "wb") as target:
                    shutil.copyfileobj(source, target)


def bbox_vertices_to_coco_bbox(row: dict, image_width: int, image_height: int) -> list[float]:
    xs = [float(row[f"bboxx{i} [pixel]"]) for i in range(1, 5)]
    ys = [float(row[f"bboxy{i} [pixel]"]) for i in range(1, 5)]

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(image_width), max(xs))
    y_max = min(float(image_height), max(ys))

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_obsea_pangaea_to_coco(
    table_path: Path,
    image_archive_path: Path,
    coco_images_path: Path,
    coco_annotations_path: Path,
) -> Path:
    """
    Converts the original OBSEA PANGAEA annotation table into COCO format.
    """
    rows = read_pangaea_table(table_path)
    image_filenames = sorted({row["IMAGE"] for row in rows})
    species_names = sorted({row["Species"] for row in rows})
    species_to_category_id = {
        species_name: i + 1 for i, species_name in enumerate(species_names)
    }

    print(f"Unique images in table: {len(image_filenames)}")
    print(f"Total annotations before filtering: {len(rows)}")
    print(f"Species/categories before compression: {len(species_names)}")

    extract_images_from_archive(image_archive_path, image_filenames, coco_images_path)

    image_id_by_filename = {}
    image_info_by_filename = {}

    for image_id, image_filename in enumerate(image_filenames, start=1):
        image_path = coco_images_path / image_filename
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        image_id_by_filename[image_filename] = image_id
        image_info_by_filename[image_filename] = {
            "id": image_id,
            "file_name": image_filename,
            "width": image_width,
            "height": image_height,
            "source_month": None,
        }

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": category_id, "name": species_name}
            for species_name, category_id in species_to_category_id.items()
        ],
    }

    annotation_id = 1
    skipped_empty_boxes = 0

    for row in rows:
        image_filename = row["IMAGE"]
        image_info = image_info_by_filename[image_filename]
        image_info["source_month"] = row["Date/Time"][:7]

        bbox = bbox_vertices_to_coco_bbox(
            row,
            image_info["width"],
            image_info["height"],
        )

        if bbox[2] <= 0 or bbox[3] <= 0:
            skipped_empty_boxes += 1
            continue

        coco_data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id_by_filename[image_filename],
                "category_id": species_to_category_id[row["Species"]],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    coco_data["images"] = list(image_info_by_filename.values())

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Saved COCO annotations to {coco_annotations_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(coco_data['categories'])}")
    if skipped_empty_boxes:
        print(f"Skipped empty boxes after clipping: {skipped_empty_boxes}")

    return coco_annotations_path


def main():
    raw_data_path = settings.raw_dir / "obsea_pangaea"
    table_path = raw_data_path / PANGAEA_TABLE_NAME
    image_archive_path = raw_data_path / PANGAEA_IMAGE_ARCHIVE_NAME

    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_annotations_path = processing_dir / settings.coco_file_name

    if coco_images_path.exists():
        shutil.rmtree(coco_images_path)
    if coco_annotations_path.exists():
        coco_annotations_path.unlink()

    convert_obsea_pangaea_to_coco(
        table_path,
        image_archive_path,
        coco_images_path,
        coco_annotations_path,
    )

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    if compressed_annotations_path.exists():
        compressed_annotations_path.unlink()

    compress_annotations_to_single_category(
        coco_annotations_path,
        None,
        compressed_annotations_path,
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path,
        compressed_annotations_path,
        DATASET_SHORTNAME,
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    from datasets.utils.visualization import save_preview_image

    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    with open(compressed_annotations_path, "r") as f:
        coco_data = json.load(f)

    source_month_by_filename = {
        image["file_name"]: image.get("source_month") for image in coco_data["images"]
    }

    def should_the_image_be_included_in_train_set(image_filename: str) -> bool:
        return source_month_by_filename[image_filename] not in VALIDATION_MONTHS

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )

    if train_dataset_path.exists():
        shutil.rmtree(train_dataset_path)
    if val_dataset_path.exists():
        shutil.rmtree(val_dataset_path)

    train_dataset_path.mkdir(parents=True, exist_ok=True)
    val_dataset_path.mkdir(parents=True, exist_ok=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
