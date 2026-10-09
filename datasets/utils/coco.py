import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

from datasets.settings import Settings


def compress_annotations_to_single_category(
    annotations_path: Path, categories_filter: Optional[List[str]], output_path: Path
):
    """
    Discards all annotations except for the ones in the categories_filter list.
    For the ones that are kept, it renames all categories to a single category, fish.

    NOTE: If categories_filter is None, all annotations are kept.

    DEPRECATED: kept only for scripts not yet migrated to
    map_annotations_to_fish_and_non_fish, which also keeps non-fish annotations.
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
            annotation["category_id"] = Settings.fish_category_id
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


def map_annotations_to_fish_and_non_fish(
    annotations_path: Path, categories_mapping: Dict[str, str], output_path: Path
):
    """
    Maps every source category to "fish" or "non-fish" according to categories_mapping
    ({source category name: "fish" | "non-fish"}). Annotations whose category is not
    in categories_mapping are discarded.
    """
    if output_path.exists():
        print(f"New annotation file already exists at {output_path}")
        return output_path

    invalid_targets = set(categories_mapping.values()) - set(Settings.category_name_to_id)
    assert not invalid_targets, f"Invalid target categories: {invalid_targets}"

    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    source_category_names = {c["id"]: c["name"] for c in coco_data["categories"]}
    missing = set(categories_mapping) - set(source_category_names.values())
    if missing:
        print(f"WARNING: categories in mapping but not in source: {sorted(missing)}")

    new_annotations = []
    source_counts = Counter()
    for annotation in coco_data["annotations"]:
        source_name = source_category_names[annotation["category_id"]]
        source_counts[source_name] += 1
        target_name = categories_mapping.get(source_name)
        if target_name is not None:
            annotation["category_id"] = Settings.category_name_to_id[target_name]
            new_annotations.append(annotation)

    print("Category mapping (source -> target: count):")
    for source_name, count in source_counts.most_common():
        target_name = categories_mapping.get(source_name, "discard")
        print(f"  {source_name} -> {target_name}: {count}")
    target_counts = Counter()
    for source_name, count in source_counts.items():
        if source_name in categories_mapping:
            target_counts[categories_mapping[source_name]] += count
    print(
        f"Kept {len(new_annotations)} of {len(coco_data['annotations'])} annotations: "
        f"{dict(target_counts)}"
    )

    coco_data["categories"] = Settings.coco_categories
    coco_data["annotations"] = new_annotations

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
