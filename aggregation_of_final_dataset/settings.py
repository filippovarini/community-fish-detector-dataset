from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    # Path of the FISH-DATASETS folder 
    personal_root_path: Path = Path("~/fish-datasets").expanduser() 
    
    data_path: Path = personal_root_path / "data"

    raw_dir: Path =  data_path / "raw"
    intermediate_dir: Path = data_path / "processing"
    processed_dir: Path = data_path / "final"

    train_dataset_suffix: str = "_train"
    val_dataset_suffix: str = "_val"
    images_folder_name: str = "JPEGImages"

    # We only use one category for the fish
    coco_category_id: int = 1
    coco_categories = [{"id": coco_category_id, "name": "fish"}]
    coco_file_name: str = "annotations_coco.json"
    
    # AI
    train_val_split_ratio: float = 0.2
    random_state: int = 42
    
    