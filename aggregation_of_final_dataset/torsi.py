## The dataset can't be downloaded programmatically.
## Please download it from the following link:
## https://data.csiro.au/collection/64913
## 
## Once Downloaded, put the .zip file in the folder fish-datasets/data/raw/torsi (create the folder if not exists)

import json
from pathlib import Path
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    copy_images_to_processing, 
    add_dataset_shortname_to_image_names
)
from data_preview.visualize_torsi import download_data

settings = Settings()
DATASET_SHORTNAME = "torsi"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = settings.raw_dir / DATASET_SHORTNAME / "data" / "instances.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"




def adjust_path():
    # remove relative path from json leaving only the name of the images
    with open(compressed_annotations_path, 'r', encoding='utf-8') as annotations_file:
        annotations_json = json.load(annotations_file)
    for image in annotations_json["images"]:
        old_filename = image["file_name"]
        new_filename = old_filename.split("/")[2]
        image["file_name"] = new_filename
    
    with open(compressed_annotations_path, 'w', encoding='utf-8') as annotations_file:
        json.dump(annotations_json, annotations_file, indent=2)
  

def processing():
    processing_dir.mkdir(parents=True)

    # take all the images from the dataset and put them into JPEGImages
    images_path.mkdir(parents=True)
    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME / "data/images/port")
    

    # Keep only fish categories
    categories_filter = ["orange_roughy", "eel", "misc_fish", "orange_roughy_edge", "chimera",
                         "oreo", "shark", "whiptail"]
    compress_annotations_to_single_category(
        annotations_path, categories_filter, compressed_annotations_path
    )

    adjust_path()

    add_dataset_shortname_to_image_names(DATASET_SHORTNAME, images_path, compressed_annotations_path)


      

def dataset_splitting():
    # Split based on days: 
    #   - Images taken on 2019-07-13, 2019-07-14 and 2019-07-15 are in the train set. 
    #   - Images taken on 2019-07-16 and 2019-07-17 are in the val set. 
    # Resulting split ratio is 0.19

    train_set_image_prefix = ["torsi_20190713", "torsi_20190714", "torsi_20190715"]

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).stem.split("-")[0] in train_set_image_prefix
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


def count_images_per_day():
    #count images per day (to find a splitting logic which satisfy the splitting ratio)
    filenames = [f.name for f in images_path.iterdir() if f.is_file()]
    images_per_day = {}
    for name in filenames:
        day = name.split("-")[0]
        if day not in images_per_day.keys():
            images_per_day[day] = 0
        images_per_day[day] = images_per_day[day] + 1
    print(images_per_day)
    


def main():
    
    download_data()

    processing()

    # count_images_per_day() #count images per day (to find a splitting logic which satisfy the splitting ratio)
    # The output of the previous function is:
    # {'20190714': 261, '20190716': 111, '20190715': 423, '20190717': 91, '20190713': 165}
    # Splitting logic: 
    #   train = 20190713, 20190714, 20190715
    #   eval = 20190716, 20190717

    dataset_splitting()


    
    


if __name__ == "__main__":
    main()

    

    

