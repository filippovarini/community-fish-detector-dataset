from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    base_dir = Path("./data")
    raw_dir: Path = base_dir / "raw"
    processed_dir: Path = base_dir / "final"
    intermediate_dir: Path = base_dir / "processing"
    preview_dir: Path = Path(__file__).resolve().parent.parent / "previews"

    train_dataset_suffix: str = "_train"
    val_dataset_suffix: str = "_val"
    images_folder_name: str = "JPEGImages"

    # Every source category is mapped to "fish", "non-fish", or discarded
    fish_category_id: int = 1
    non_fish_category_id: int = 2
    coco_categories = [
        {"id": fish_category_id, "name": "fish"},
        {"id": non_fish_category_id, "name": "non-fish"},
    ]
    category_name_to_id = {c["name"]: c["id"] for c in coco_categories}
    coco_file_name: str = "annotations_coco.json"

    # AI
    train_val_split_ratio: float = 0.2
    random_state: int = 42
