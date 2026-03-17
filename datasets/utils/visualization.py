import random
from pathlib import Path

import matplotlib.pyplot as plt
import supervision as sv


def get_annotation_count_from_supervision_dataset(dataset):
    # Method 1: Count all annotations across all images
    total_annotations = 0
    for image_name, annotations_list in dataset.annotations.items():
        total_annotations += len(annotations_list)

    return total_annotations


def visualize_supervision_dataset(
    dataset, num_samples=16, grid_size=(4, 4), size=(20, 12)
):
    """Visualize random samples from a dataset with bounding boxes and labels."""
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")
    print(
        f"Dataset annotation count: {get_annotation_count_from_supervision_dataset(dataset)}"
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    image_example = None
    annotated_images = []
    image_names = []

    for _ in range(num_samples):
        i = random.randint(0, len(dataset) - 1)  # Avoid index out of range

        image_path, image, annotations = dataset[i]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]

        # Get image name
        image_name = Path(image_path).stem
        image_names.append(image_name)

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)

        if len(annotations) > 0 and image_example is None:
            image_example = annotated_image

    sv.plot_images_grid(
        annotated_images,
        grid_size=grid_size,
        titles=image_names,
        size=size,
        cmap="gray",
    )

    return image_example


def build_and_visualize_supervision_dataset_from_coco_dataset(
    images_dir: Path, annotations_path: Path
):
    """
    Given the path to COCO annotations and images, build a Supervision dataset and visualize it.
    """
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=images_dir,
        annotations_path=annotations_path,
    )

    image_example = visualize_supervision_dataset(dataset)
    return image_example


def save_preview_image(images_path: Path, annotations_path: Path, shortname: str):
    """Save a sample annotated image from the dataset to the previews/ directory."""
    from datasets.settings import Settings

    settings = Settings()
    settings.preview_dir.mkdir(parents=True, exist_ok=True)

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )

    image_example = visualize_supervision_dataset(dataset)

    if image_example is not None:
        output_path = settings.preview_dir / f"{shortname}_sample_image.png"
        plt.imsave(str(output_path), image_example)
        print(f"Sample image saved to {output_path}")
    else:
        print("No annotated images found to save as sample")
