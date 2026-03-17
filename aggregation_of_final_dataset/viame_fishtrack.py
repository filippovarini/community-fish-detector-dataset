from pathlib import Path

from data_preview.visualize_viame_fishtrack import (
    download_data_and_build_coco_dataset,
    DATASET_SHORTNAME,
)
from settings import Settings

settings = Settings()


def setup_raw_processed_directories_for_dataset(dataset_name: str) -> tuple[Path, Path]:
    """
    Sets up the raw and processed directories for a dataset.
    """
    raw_data_path = settings.raw_dir / dataset_name
    processed_data_path = settings.final_dir / dataset_name
    return raw_data_path, processed_data_path


def main():
    """
    Downloads the VIAME FishTrack data.
    No need to split in train and val, as the VIAME FishTrack data is already split.
    No need to compress the annotations into fish only, as the
    download_data_and_build_coco_dataset function already does this.
    """
    # Download the train data
    train_data_name = f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    train_raw_data_path, train_coco_dataset_path = (
        setup_raw_processed_directories_for_dataset(train_data_name)
    )

    train_data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a19f85cf5a99794ea9ccfb%22,%2265a1a15fcf5a99794eaaa790%22,%2265a1a028cf5a99794eaa2419%22,%2265a19f70cf5a99794ea9c1f7%22,%2265a19f59cf5a99794ea9b5b4%22,%2265a19f70cf5a99794ea9c20c%22,%2265a1a160cf5a99794eaaa7e7%22,%2265a1a123cf5a99794eaa925a%22,%2265a19f85cf5a99794ea9cd00%22,%2265a1a040cf5a99794eaa3185%22,%2265a19f9bcf5a99794ea9d8e3%22,%2265a1a13acf5a99794eaa9c17%22,%2265a1a16dcf5a99794eaaabd2%22,%2265a1a160cf5a99794eaaa7db%22,%2265a1a162cf5a99794eaaa858%22,%2265a1a11bcf5a99794eaa8dbb%22,%2265a19f83cf5a99794ea9cc04%22,%2265a19fcecf5a99794ea9f433%22,%2265a1a144cf5a99794eaa9f0c%22,%2265a1a0dccf5a99794eaa7ac8%22]"
    download_data_and_build_coco_dataset(
        raw_data_download_path=train_raw_data_path,
        coco_dataset_path=train_coco_dataset_path,
        data_url=train_data_url,
    )

    # Download the val data
    val_data_name = f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    val_raw_data_path, val_coco_dataset_path = (
        setup_raw_processed_directories_for_dataset(val_data_name)
    )

    val_data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a1a1d1cf5a99794eaacb57%22,%2265a1a291cf5a99794eab01fb%22,%2265a1a205cf5a99794eaadbb6%22,%2265a1a223cf5a99794eaae509%22,%2265a1a20ccf5a99794eaadddd%22,%2265a1a1d1cf5a99794eaacb3d%22,%2265a1a23ecf5a99794eaaed79%22,%2265a1a20ccf5a99794eaadde0%22,%2265a1a223cf5a99794eaae50e%22,%2265a1a1d1cf5a99794eaacb52%22,%2265a1a28fcf5a99794eab01b2%22,%2265a1a22fcf5a99794eaae8c1%22,%2265a1a205cf5a99794eaadbbb%22,%2265a1a1ffcf5a99794eaad9c8%22,%2265a1a1d8cf5a99794eaacd93%22,%2265a1a1f1cf5a99794eaad548%22,%2265a1a1d1cf5a99794eaacb67%22,%2265a1a23ecf5a99794eaaed82%22,%2265a1a230cf5a99794eaae92a%22,%2265a1a244cf5a99794eaaef6b%22]"
    download_data_and_build_coco_dataset(
        raw_data_download_path=val_raw_data_path,
        coco_dataset_path=val_coco_dataset_path,
        data_url=val_data_url,
    )


if __name__ == "__main__":
    main()
