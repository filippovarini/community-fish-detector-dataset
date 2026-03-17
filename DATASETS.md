# Dataset Processing Details

Each dataset script in `datasets/` follows a 4-step pattern: Download -> Process -> Preview -> Split.

## Complete Datasets (11)

### brackish
- **Source**: [Roboflow](https://public.roboflow.com/ds/vGBLxigwno?key=bhFPGoB3VB)
- **Download**: Automatic
- **Annotations**: COCO format (from Roboflow train/val/test splits merged into one)
- **Category filter**: `small_fish`, `fish`
- **Split**: By deployment site (extracted from image filename prefix before `_jpg`)

### deep_vision
- **Source**: [NMDC FTP](https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip)
- **Download**: Automatic
- **Annotations**: CSV -> COCO conversion
- **Category filter**: None (all are fish)
- **Split**: By deployment (first token of filename before `_`)

### deepfish
- **Source**: [QLD Data](http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar)
- **Download**: Automatic
- **Annotations**: Segmentation masks -> bounding boxes (connected component analysis)
- **Category filter**: None (only fish)
- **Split**: By deployment (first token of filename before `_`)

### fathomnet
- **Source**: [FathomNet API](https://fathomnet.org/)
- **Download**: Automatic via `fathomnet-generate` CLI (~12 hours)
- **Annotations**: COCO format from FathomNet API (fish concepts: Actinopterygii, Sarcopterygii, Chondrichthyes, Myxini descendants)
- **Category filter**: None (filtered at download)
- **Split**: Random

### fishclef
- **Source**: [Zenodo](https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1)
- **Download**: Automatic
- **Annotations**: XML -> COCO conversion, frames extracted from .flv videos
- **Category filter**: None (majority "Null" = general fish)
- **Split**: By video ID

### mit_river_herring
- **Source**: [LILA](https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip)
- **Download**: Automatic
- **Annotations**: COCO format (multiple sub-datasets joined into one)
- **Category filter**: None (only herring)
- **Split**: By location+video (first two tokens of filename)

### noaa_puget
- **Source**: [LILA](https://storage.googleapis.com/public-datasets-lila/noaa-psnf/)
- **Download**: Automatic (separate image and annotation downloads)
- **Annotations**: COCO format (annotations cleaned to remove empty bboxes)
- **Category filter**: `fish`
- **Split**: By camera (third token of filename)

### project_natick
- **Source**: [GitHub](https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip)
- **Download**: Automatic
- **Annotations**: Pascal VOC -> COCO conversion (via supervision library)
- **Category filter**: `Fish`, `Squid`
- **Split**: Random (all images from same camera/location/datetime)

### roboflow_fish
- **Source**: [Roboflow](https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6)
- **Download**: Automatic
- **Annotations**: COCO format (from Roboflow train/val/test splits merged into one)
- **Category filter**: None (all are fish)
- **Split**: Pre-provided (uses Roboflow's original split)

### viame_fishtrack
- **Source**: [VIAME](https://viame.kitware.com/)
- **Download**: Automatic (separate train/val URLs)
- **Annotations**: VIAME CSV -> COCO conversion, frames extracted from videos
- **Category filter**: Non-fish excluded at download time
- **Split**: Pre-split by URL (train and val downloaded separately)

### zebrafish
- **Source**: [Kaggle](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)
- **Download**: Automatic via kagglehub
- **Annotations**: CSV -> DataFrame -> COCO conversion
- **Category filter**: None (only Zebrafish)
- **Split**: By video (Vid1 -> train, Vid2 -> val, hardcoded)

## Partial Datasets (5)

These datasets require manual download or have special dependencies.

### coralscapes
- **Source**: [Coralscapes](https://josauder.github.io/coralscapes/)
- **Download**: Manual - download Parquet files and place in `data/raw/coralscapes/`
- **Annotations**: Parquet -> segmentation masks -> bounding boxes (connected components)
- **Category filter**: fish only (category ID 9 in source, remapped to 1)
- **Split**: By site (sites 1-24 train, sites 25-35 val)
- **Special**: Requires `pyarrow` for Parquet reading

### torsi
- **Source**: [CSIRO](https://data.csiro.au/collection/64913)
- **Download**: Manual - download zip and place in `data/raw/torsi/`
- **Annotations**: COCO format (already provided)
- **Category filter**: `orange_roughy`, `eel`, `misc_fish`, `orange_roughy_edge`, `chimera`, `oreo`, `shark`, `whiptail`
- **Split**: By date (2019-07-13/14/15 train, 2019-07-16/17 val, ~19% val ratio)

### f4k
- **Source**: [UniCT OneDrive](https://studentiunict-my.sharepoint.com/...)
- **Download**: Manual - download zip and place in `data/raw/f4k/`
- **Annotations**: XML contour annotations -> bounding boxes, frames extracted from .mp4 videos
- **Category filter**: `fish`
- **Split**: By video ID (106-109 val, 110-124 train)
- **Special**: Requires `ffmpeg` for video frame extraction

### kakadu
- **Source**: Manual download
- **Download**: Manual - place dataset in `data/raw/kakadu/`
- **Annotations**: COCO format (cleaned to remove empty bboxes)
- **Category filter**: None (all are fish)
- **Split**: Random

### marine_detect
- **Source**: Roboflow (two datasets: FishInv + Megafauna)
- **Download**: Manual - download both datasets from Roboflow and place in `data/raw/marine_detect/`
- **Annotations**: YOLO text -> COCO conversion, two datasets merged
- **Category filter**: `turtle`, `ray`, `shark`, `bolbometopon_muricatum`, `chaetodontidae`, `cheilinus_undulatus`, `cromileptes_altivelis`, `fish`, `haemulidae`, `lutjanidae`, `muraenidae`, `scaridae`, `serranidae`
- **Split**: By original split suffix in filename (train/valid/test)
- **Special**: OzFish images are skipped to avoid duplication
