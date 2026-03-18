import json
from pathlib import Path
from typing import List, Optional

from datasets.settings import Settings


def compress_annotations_to_single_category(
    annotations_path: Path, categories_filter: Optional[List[str]], output_path: Path
):
    """
    Discards all annotations except for the ones in the categories_filter list.
    For the ones that are kept, it renames all categories to a single category, fish.

    NOTE: If categories_filter is None, all annotations are kept.
    """
    # Check if new annotation file already exists
    if output_path.exists():
        print(f"New annotation file already exists at {output_path}")
        return output_path

    # Load the annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Check existing categories
    all_category_names = {c["name"] for c in coco_data["categories"]}
    if categories_filter is None:
        print(f"Found categories {all_category_names} but keeping all")
    else:
        print(
            f"Found categories {all_category_names} but only keeping {categories_filter}"
        )

    # Filter annotations to only include the ones in the categories_filter list
    new_annotations = []
    for annotation in coco_data["annotations"]:
        # Annotation ids in COCO are 1-indexed but list indices are 0-indexed
        category_index = annotation["category_id"] - 1
        annotation_category = coco_data["categories"][category_index]

        assert (
            annotation["category_id"] == annotation_category["id"]
        ), f"Annotation category_id is {annotation['category_id']} not {annotation_category['id']}"

        if not categories_filter or annotation_category["name"] in categories_filter:
            annotation["category_id"] = Settings.coco_category_id
            new_annotations.append(annotation)

    # Print the number of annotations before and after compression
    original_annotation_count = len(coco_data["annotations"])
    compressed_annotation_count = len(new_annotations)
    print(f"Original annotation count: {original_annotation_count}")
    print(f"Compressed annotation count: {compressed_annotation_count}")

    # Compress categories to a single category
    coco_data["categories"] = Settings.coco_categories
    coco_data["annotations"] = new_annotations

    # Store the new annotation file
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    return output_path


def convert_coco_annotations_from_0_indexed_to_1_indexed(
    input_coco_annotations_path: Path, output_coco_annotations_path: Path
) -> dict:
    """
    The standard COCO categories should be 1-indexed but some datasets are 0-indexed.
    This function converts the category ids to 1-indexed.
    """
    if output_coco_annotations_path.exists():
        print(f"New annotation file already exists at {output_coco_annotations_path}")
        return output_coco_annotations_path

    with open(input_coco_annotations_path, "r") as f:
        coco_data = json.load(f)

    for annotation in coco_data["annotations"]:
        annotation["category_id"] += 1
    for category in coco_data["categories"]:
        category["id"] += 1

    with open(output_coco_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    return output_coco_annotations_path
