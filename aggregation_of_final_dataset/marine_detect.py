# This script downloads, merges and process two datasets: Marine Detect FishInv and Marine Detect Megafauna
# These two datasets were merged as there were many common images with annotations divided between the two datasets. 

import shutil
from pathlib import Path
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    copy_images_to_processing,
    add_dataset_shortname_to_image_names
)
from data_preview.visualize_marine_detect import(
    download_data,
    create_coco_datasets,
    merge_datasets
)

settings = Settings()
DATASET_SHORTNAME = "marine_detect"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = processing_dir / "annotations.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"


def processing():
    processing_dir.mkdir(parents=True)
    create_coco_datasets()
    merge_datasets()

    # take all the images from the dataset and put them into JPEGImages
    images_path.mkdir(parents=True)
    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME / "images")
    
    shutil.copy(str(settings.raw_dir) + f"/{DATASET_SHORTNAME}/annotations.json", annotations_path)

    keep_categories = [
    "turtle", 
    "ray", 
    "shark", 
    "bolbometopon_muricatum", 
    "chaetodontidae", 
    "cheilinus_undulatus",
    "cromileptes_altivelis", 
    "fish", 
    "haemulidae", 
    "lutjanidae", 
    "muraenidae",
    "scaridae", 
    "serranidae"
    ]

    compress_annotations_to_single_category(
        annotations_path, keep_categories, compressed_annotations_path
    )

    add_dataset_shortname_to_image_names(DATASET_SHORTNAME, images_path, compressed_annotations_path)

   




def dataset_splitting():
    # Keeping original dataset split

    should_the_image_be_included_in_train_set = (
        # images have a variable number of "_" in the name. Last thing before extension is the name of the set (train, test or valid)
        # example marine_detect_megafauna__112230728_gettyimages-138058528_jpg.rf.bc2b8e44ecb9db62e5e6a194f367e647_train.jpg
        # next line extracts set name ("train" in the example)
        lambda image_path: Path(image_path).stem.split("_")[-1] == "train"
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
        images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


    

def main():

    download_data()

    processing()

    dataset_splitting()


    
    


if __name__ == "__main__":
    main()

    

    


