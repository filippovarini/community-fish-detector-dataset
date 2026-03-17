
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    get_train_images_with_random_splitting,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    copy_images_to_processing,
    add_dataset_shortname_to_image_names
)
from data_preview.visualize_kakadufishai import download_data, clean_annotations
import shutil



settings = Settings()

DATASET_SHORTNAME = "kakadu"
annotations_path = settings.intermediate_dir / DATASET_SHORTNAME / "annotations_coco.json"

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME

coco_images_path = processing_dir / settings.images_folder_name
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"

def annotations_processing():

    clean_annotations()

    processing_dir.mkdir(parents=True)

    # take all the images from the dataset and put them into JPEGImages
    coco_images_path.mkdir(parents=True)
    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME )
    

    shutil.move(str(processing_dir) + f"/JPEGImages/KakaduFishAI_boundingbox.json", annotations_path)


    # Keep all categories as all are fish
    categories_filter = None
    compress_annotations_to_single_category(
        annotations_path, categories_filter, compressed_annotations_path
    )

    add_dataset_shortname_to_image_names(DATASET_SHORTNAME, coco_images_path, compressed_annotations_path)



def dataset_splitting():
    # Random Splitting
    train_image_names = get_train_images_with_random_splitting(coco_images_path)
    should_the_image_be_included_in_train_set = (
        lambda image_path: image_path in train_image_names
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
    


if __name__=="__main__":

    download_data()

    annotations_processing()

    dataset_splitting()
