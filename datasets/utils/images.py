import json
import shutil
from pathlib import Path

import tqdm


def add_dataset_shortname_prefix_to_image_names(
    images_path: Path,
    annotations_path: Path,
    dataset_shortname: str,
) -> None:
    """
    Converts the image filenames to be f"{dataset_shortname}_{image_filename}"
    """
    # Assert all paths are valid
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found at {images_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")

    print(f"Adding dataset shortname prefix to image names: {dataset_shortname}")

    # Load the annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Add the dataset shortname prefix to the image filenames
    for image in tqdm.tqdm(coco_data["images"], total=len(coco_data["images"])):
        old_image_filename = Path(image["file_name"]).name

        if old_image_filename.startswith(dataset_shortname):
            continue

        new_image_filename = f"{dataset_shortname}_{old_image_filename}"

        # Rename the image file
        old_image_path = images_path / old_image_filename
        new_image_path = images_path / new_image_filename
        try:
            assert old_image_path.exists(), f"Image not found at {old_image_path}"
            assert not new_image_path.exists(), f"Image already exists at {new_image_path}"
        except Exception as e:
            print(f"Error renaming image {old_image_filename} to {new_image_filename}: {e}")
            continue

        old_image_path.rename(new_image_path)

        # Update the annotation with the new image filename
        image["file_name"] = new_image_filename

    # Save the annotations
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)


def remove_dataset_shortname_prefix_from_image_filename(
    image_filename: str, dataset_shortname: str
) -> str:
    """
    Removes the dataset shortname prefix from the image filenames.
    """
    return image_filename.replace(f"{dataset_shortname}_", "")


def copy_images_to_processing(
    dataset_shortname: str,
    source_dir: Path,
) -> Path:
    """
    Copies all image files from source_dir to the processing directory
    (settings.intermediate_dir / dataset_shortname / JPEGImages).
    """
    from datasets.settings import Settings

    settings = Settings()
    dest_dir = settings.intermediate_dir / dataset_shortname / settings.images_folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".json"}
    for file_path in source_dir.iterdir():
        if file_path.suffix.lower() in image_extensions:
            shutil.copy2(file_path, dest_dir / file_path.name)

    print(f"Copied images from {source_dir} to {dest_dir}")
    return dest_dir
