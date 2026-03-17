from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    base_dir = Path("/mnt/data/dev/fish-datasets/data")
    raw_dir: Path = base_dir / "raw"
    processed_dir: Path = base_dir / "final"
    intermediate_dir: Path = base_dir / "processing"
    
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
    
    

