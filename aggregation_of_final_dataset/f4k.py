f"""

This script assumes that you have downloaded the zip file.

Download link: https://studentiunict-my.sharepoint.com/personal/simone_palazzo_unict_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsimone%5Fpalazzo%5Funict%5Fit%2FDocuments%2FDatasets%5Freleased%2Ff4k%5Fdetection%5Ftracking%2Ezip&parent=%2Fpersonal%2Fsimone%5Fpalazzo%5Funict%5Fit%2FDocuments%2FDatasets%5Freleased&ga=1

Once downloaded, put it in fish-dataset/data/raw/f4k

NOTE: This script requires ffmpeg

"""
import json
import shutil
from pathlib import Path
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    copy_images_to_processing, 
    add_dataset_shortname_to_image_names,
    convert_coco_annotations_from_0_indexed_to_1_indexed
)
from data_preview.visualize_f4k import extract_data, clean_annotations


settings = Settings()
DATASET_SHORTNAME = "f4k"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = processing_dir / "annotations.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME / "f4k_detection_tracking"
input_images_dir = settings.raw_dir / DATASET_SHORTNAME / "coco"




def processing():
    processing_dir.mkdir(parents=True)
    
    if not input_images_dir.exists():
        clean_annotations()

    # take all the images (and annotations.json) from the dataset and put them into JPEGImages
    images_path.mkdir(parents=True)
    copy_images_to_processing(DATASET_SHORTNAME, input_images_dir)
    
    shutil.move(str(images_path / "annotations_coco.json"), str(annotations_path))


    # Categories indexes must be correct
    corrected_annotations_path = processing_dir / "corrected_annotations.json"
    convert_coco_annotations_from_0_indexed_to_1_indexed(annotations_path, corrected_annotations_path)

    keep_categories = ["fish"]

    compress_annotations_to_single_category(
        corrected_annotations_path, keep_categories, compressed_annotations_path
    )

    add_dataset_shortname_to_image_names(DATASET_SHORTNAME, images_path, compressed_annotations_path)

   




def dataset_splitting():
    # Splitting logic: 
    #   val = videos 106 - 109
    #   train = videos 110 - 124

    should_the_image_be_included_in_train_set = (
    # verify if the video ID is more than 109, extracting it from the filename
    # e.g. f4k_video_gt_106_frame_111.jpg -> 106 > 109
        lambda image_path: int(Path(image_path).stem.split("_")[3]) > 109
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

def count_frame_per_video():
    #count frame per video (to find a splitting logic which satisfy the splitting ratio)
    filenames = [f.name for f in images_path.iterdir() if f.is_file()]
    images_per_video = {}
    for name in filenames:
        video = name.split("_")[3]
        if video not in images_per_video.keys():
            images_per_video[video] = 0
        images_per_video[video] = images_per_video[video] + 1
    for i in range(106, 125):
        key = str(i)
        if key == "108" or key == "115":
            continue
        print("video " + key + " -> " + str(images_per_video[key]))
    
    

def main():

    if not raw_data_dir.exists():
        extract_data()

    
    processing()

    # count_frame_per_video() # count frame per video (to find a splitting logic which satisfy the splitting ratio)
    # Splitting logic: 
    #   val = videos 106 - 109
    #   train = videos 110 - 124

    dataset_splitting()


    
    


if __name__ == "__main__":
    main()

    

    


