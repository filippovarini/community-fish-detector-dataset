"""
    This script assumes that you have downloaded the Coralscapes dataset:
        https://josauder.github.io/coralscapes/    
    ...as a series of Parquet files.  

    Download the dataset in the fish-datasets/data/raw folder
    The result must be fish-datasets/data/raw/coralscapes/ ( data | figures | id2label.json | label2color.json | ...)

"""



import json
from pathlib import Path
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    copy_images_to_processing,
    add_dataset_shortname_to_image_names
)
from data_preview.visualize_coralscapes import create_coco_dataset


settings = Settings()
DATASET_SHORTNAME = "coralscapes"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = settings.raw_dir / DATASET_SHORTNAME / "coco/coralscapes-coco/coralscapes.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"


def processing():
    """ This function performs usual processing and also resolves these issues:
            - There is only fish category but has ID=9. It must be ID=1
            - Images ID and annotations ID are strings and not integers. 
            - Missing fields area and iscrowd in annotation must be added
            
            """
    
    processing_dir.mkdir(parents=True)

    # take all the images from the dataset and put them into JPEGImages
    images_path.mkdir(parents=True)
    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME / "coco/coralscapes-coco")
    
    # resolve issues mentioned above
    with open(annotations_path, 'r', encoding='utf-8') as annotations_file:
        annotations_json = json.load(annotations_file)

    image_id = 1
    annotation_id = 1

    for image in annotations_json["images"]:
        old_id = image["id"]
        image["id"] = image_id
        
        for annotation in annotations_json["annotations"]:
            if annotation["image_id"] == old_id:
                bbox = annotation["bbox"]

                annotation["id"] = annotation_id
                annotation["image_id"] = image_id
                annotation["category_id"] = 1
                annotation["area"] = bbox[2] * bbox[3]
                annotation["iscrowd"] = 0

                annotation_id = annotation_id + 1

        image_id = image_id +1
    
    annotations_json["categories"][0]["id"] = 1
    
    with open(compressed_annotations_path, 'w', encoding='utf-8') as annotations_file:
        json.dump(annotations_json, annotations_file, indent=2)

    add_dataset_shortname_to_image_names(DATASET_SHORTNAME, images_path, compressed_annotations_path)


    

def dataset_splitting():
    # Build Logic to split into train and val based on camera name
    
    should_the_image_be_included_in_train_set = (
        # verify if the number of the site is less than 25, extracting it from the filename, e.g. coralscapes_site23_xyz.png -> 23 < 25
        lambda image_path: int(Path(image_path).stem.split("_")[1].split("e")[1]) < 25 
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


def count_images_per_site():
    #count images per day (to find a splitting logic which satisfy the splitting ratio)
    filenames = [f.name for f in images_path.iterdir() if f.is_file()]
    images_per_site = {}
    for name in filenames:
        site = name.split("_")[0]
        if site not in images_per_site.keys():
            images_per_site[site] = 0
        images_per_site[site] = images_per_site[site] + 1
    for i in range(1, 36):
        key = "site" + str(i)
        print(key + " -> " + str(images_per_site[key]))
    

def main():

    create_coco_dataset()

    processing()


    # count_images_per_site() # count images per site/location (to find a splitting logic which satisfy the splitting ratio)
    # Splitting logic: 
    #   train = sites 1-24
    #   eval = sites 25-35

    dataset_splitting()
    
    
    


if __name__ == "__main__":
    main()

    

    
