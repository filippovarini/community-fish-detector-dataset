# Dataset Processing Details

Each dataset script in `datasets/` follows a 4-step pattern: Download -> Process -> Preview -> Split.

## Complete Datasets

### brackish
- **Source**: [Roboflow](https://public.roboflow.com/ds/vGBLxigwno?key=bhFPGoB3VB)
- **Download**: Automatic
- **Annotations**: COCO format (from Roboflow train/val/test splits merged into one)
- **Category mapping**:
  - fish: `fish`, `small_fish`
  - non-fish: `crab`, `starfish`, `jellyfish`, `shrimp`
  - discard: `animals` (Roboflow supercategory, no annotations)
- **Split**: By deployment site (extracted from image filename prefix before `_jpg`)

### coralscapes_v2
- **Source**: [CoralscapesV2](https://josauder.github.io/coralscapesv2/) ([Hugging Face](https://huggingface.co/datasets/josauder/coralscapesV2), `instances` configuration)
- **Download**: Automatic (13 Parquet shards, ~6.9 GB). Each row holds the image, the 95-class semantic mask, the fish instance ids and the site. The `sequences` configuration (31-frame video clips, ~206 GB) is not used. Replaces Coralscapes v1: CoralscapesV2 contains all v1 images but one
- **Annotations**: Segmentation -> boxes. Fish: one box per instance mask (fish ids are `25 * 2000 + index`). Non-fish classes have no instance masks: one box per connected component of the semantic mask, ignoring components under 200 px (mostly slivers of animals occluded by coral). 2,433 images (2048x1024 PNG) from 41 Red Sea sites; 211 images have no fish or non-fish box and are kept
- **Category mapping**:
  - fish: `fish` (65,738 instances; many are small: median box long side 22 px)
  - non-fish: `acanthaster planci` (crown-of-thorns starfish), `sea cucumber`, `other animal` (mostly sea turtles, some starfish)
  - discard: `sea urchin` (urchins cluster and have no instance masks, so one component often covers several urchins), `giant clam`, `dead giant clam`, `anemone`, `feather worm`, `cirrhipathes`, `sponge`, `human`, `human-made structure`, `transect line`, `transect tools`, `trash`, and all coral, algae, substrate, seagrass, water and background classes
- **Split**: By site. The sites in the authors' validation and test splits (site1, site6, site22, site23, site26, site29, site34) go to val; site30, which is in both their train and validation splits, goes to train (~21% of images and fish in val)

### deep_vision
- **Source**: [NMDC FTP](https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip)
- **Download**: Automatic
- **Annotations**: CSV -> COCO conversion. Reads the four per-year CSVs plus the root-level `test_annotations.csv` / `val_annotations.csv`; an image takes its boxes from the first CSV that lists it (4 images are annotated in two CSVs). Some rows give the wrong species folder (e.g. `test/mackerel/` for a file in `test/mixed/`); those images are found by filename. One listed image (`20170511143523664.jpg`) does not exist and is skipped
- **Category mapping**:
  - fish: herring, mackerel, bluewhiting, mesopelagic (lanternfish-type fish), Sharpchin barracudina, spotted barracudina, salmon, lumpsucker
  - non-fish: none
  - discard: none
- **Split**: By trawl station (first token of filename split on `-` or `_`, e.g. `ST019`, `ST7`; 40 stations)

### deepfish
- **Source**: [QLD Data](http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar)
- **Download**: Automatic
- **Annotations**: Segmentation masks -> bounding boxes (connected component analysis). Uses the 310 `valid` frames (with fish) and the 310 `empty` frames (no fish, kept as background)
- **Category mapping**:
  - fish: `fish` (every mask component of at least 100 px)
  - non-fish: none
  - discard: mask noise (14 components of 1-20 px)
- **Split**: By habitat (first token of filename before `_`, 19 habitats). Habitats with fish (13) and habitats with only empty frames (6) are split separately so val is not mostly empty

### fathomnet
- **Source**: [FathomNet API](https://fathomnet.org/)
- **Download**: Automatic. The image records of every fish and non-fish concept are fetched from the FathomNet API and written as COCO annotations (boxes of discarded concepts are dropped, images without a fish or non-fish box are not fetched), then the images are downloaded from their URLs. `fathomnet-generate` is not used: it crashes on images with no width/height in FathomNet, which are filled in from the downloaded files. Takes many hours; an interrupted run resumes, skipping images already on disk
- **Annotations**: COCO format from the FathomNet API
- **Category mapping**: every FathomNet concept is assigned from its WoRMS lineage (deepest matching rank wins; for `X sp.` / `X complex` labels the genus is looked up), plus a hand-written list for common names with no WoRMS match. The resolved concept -> class mapping is committed in [`datasets/fathomnet_categories.json`](datasets/fathomnet_categories.json) (rebuild with `python -m datasets.fathomnet --rebuild-categories`). Box counts from the API on 2026-10-11:
  - fish (698k boxes, 714 concepts): Actinopterygii, Sarcopterygii, Chondrichthyes, Agnatha; `bony fish`, `shark`, `ray or skate`, `chimaera`, `vertebrate` and other unmatched fish labels
  - non-fish (214k boxes, 784 concepts): Tetrapoda; Asterozoa, Echinoidea, Holothuroidea (except Psolidae), Comatulida / `feather star`; Decapoda, Stomatopoda, Lophogastrida, Aegidae, Cirolanidae, Colossendeidae; Cephalopoda, Neogastropoda, Ringipleura, Pseudothecosomata, Pterotracheoidea, Seguenziida, Pleurotomariida; Scyphozoa, Staurozoa, Cubozoa, Narcomedusae (except *Solmaris*, *Solmundella*), *Colobonema*, *Halicreas*, *Halitrephes*, *Mitrocoma*; Lobata, Beroida; `brittle star`, `urchin`, `sea star`, `sea cucumber`, `shrimp`, `crab`, `squid`, `octopus`, `sea turtle` and other unmatched labels
  - discard (385k boxes, 1,436 concepts): corals, sea pens, zoanthids, hydroids, sponges, anemones, corallimorphs, sea squirts, brachiopods, bryozoans, barnacles, bivalves, all worms and tunicates, siphonophores, chaetognaths, small or generic jellies, cydippid and generic ctenophores, attached sea cucumbers (Psolidae), stalked crinoids, limpets and small snails, amphipods, mysids, krill, copepods, most isopods, generic labels (`marine organism`, `Chordata`, `echinoderm`, `crustacean`, ...), `Holothuroidea/Actiniaria`, protists, algae and equipment
- **Split**: Random by image (sorted file names, `random_state`)

### fishclef
- **Source**: [Zenodo](https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1)
- **Download**: Automatic
- **Annotations**: XML -> COCO conversion, frames extracted from .flv videos
- **Category mapping**:
  - fish: all 29 species names (reef fish), including `NULL` (unidentified fish)
  - non-fish: none
  - discard: none
- **Species attribute**: `fish_species` in the training set XMLs, `species_name` in the test set XMLs (both read)
- **Split**: By video ID

### mit_river_herring
- **Source**: [LILA](https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip)
- **Download**: Automatic
- **Annotations**: COCO format (multiple sub-datasets joined into one)
- **Category mapping**:
  - fish: river_herring, trout, perch, bass, eel, lamprey, sunfish, pickerel
  - non-fish: none
  - discard: unknown_fish. Images containing it are dropped entirely (2,734 frames), as these boxes mix real fish with lamp glare, debris, bubbles and the timestamp overlay
- **Empty frames**: kept (~77% of frames have no boxes)
- **Split**: By location+video (first two tokens of filename)

### noaa_puget
- **Source**: [LILA](https://storage.googleapis.com/public-datasets-lila/noaa-psnf/)
- **Download**: Automatic (separate image and annotation downloads)
- **Annotations**: COCO format (images containing `fish_or_crab` / `unknown` boxes are dropped entirely)
- **Category mapping**:
  - fish: `fish`
  - non-fish: `crab`
  - discard: `empty` (no-animal marker, no bbox), `fish_or_crab`, `unknown`
- **Split**: By camera (`SDxx` camera ID, first token of filename)

### project_natick
- **Source**: [GitHub](https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip)
- **Download**: Automatic
- **Annotations**: Pascal VOC -> COCO conversion. Both subsets are used: `fish_arrow_worms_annotation` (1,072 frames, one video) and `fish_annotations` (46 frames, two videos)
- **Category mapping**:
  - fish: `Fish`
  - non-fish: none
  - discard: `Squid`. Despite the label these are arrow worms (chaetognaths), thin translucent planktonic worms mostly under ~3 cm
- **Split**: By chunks of 25 consecutive frames per video (49 chunks). There are only 3 videos, one with 96% of the frames, so a per-video split is not possible

### roboflow_fish
- **Source**: [Roboflow](https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6)
- **Download**: Automatic
- **Annotations**: COCO format (from Roboflow train/val/test splits merged into one). The export holds 670 of the 680 source images twice (pixel-identical, identical boxes, one copy with Latin family names and one with common names); one copy is kept, by source filename (the part before `.rf.<hash>`)
- **Category mapping**:
  - fish: all 26 source classes (Latin family names, e.g. `Scombridae -Tunas-`, and common names, e.g. `tuna`) plus Roboflow's parent category `fish`
  - non-fish: none
  - discard: none
- **Split**: Pre-provided (uses Roboflow's original split)

### viame_fishtrack
- **Source**: [VIAME](https://viame.kitware.com/)
- **Download**: Automatic (separate train/val URLs)
- **Annotations**: VIAME CSV -> COCO conversion, frames extracted from videos
- **Category mapping**:
  - fish: `micropterus_salmoides`, `unspecified_fish`, `etelis_coruscans`, `hyporthodus_quernus`, `pristipomoides_zonatus`, `pristipomoides_auricilla`, `seriola`, `caranx_melampygus`
  - non-fish: `non_fish_animal`
  - discard: `non_fish_bait`, `non_fish_plant`
- **Split**: Pre-split by URL (train and val downloaded separately)

### zebrafish
- **Source**: [Kaggle](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)
- **Download**: Automatic via kagglehub
- **Annotations**: CSV -> DataFrame -> COCO conversion (2,224 frames, 3 boxes per frame)
- **Category mapping**:
  - fish: `Zebrafish` (6,672 boxes, including those flagged as occluded or glitched)
  - non-fish: none
  - discard: none
- **Split**: By video (Vid1 -> train, Vid2 -> val, hardcoded). Only two videos of the same tank, so any finer split would put near-identical frames in both sets

### orange_chromide
- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/7w45jx35hd/1)
- **Download**: Automatic (17 MB zip)
- **Annotations**: YOLO TXT -> COCO conversion (single unnamed class, 640x640 images; source train/valid/test merged). Images are Roboflow exports of pond video frames (`frame<n>_jpg.rf.<hash>.jpg`)
- **Category mapping**:
  - fish: the single class (every fish is boxed, not only orange chromides: other cichlids and koi too)
  - non-fish: none
  - discard: none
- **Split**: By blocks of 100 frame numbers (`frame<n> // 100`, 26 blocks; sorted, `random_state`). Consecutive frames are near-duplicates, so a random split would put neighbouring frames in train and val. The source video is not in the filename and frame numbers restart per video (31 numbers appear twice, from different scenes); a block can mix frames of different videos, but neighbouring frames of any video stay together

### pomerfish
- **Source**: [Zenodo](https://zenodo.org/records/17432128)
- **Download**: Automatic (`PomerFish.rar`, 25.1 GB via `unar`)
- **Annotations**: COCO format (already provided, 1-indexed)
- **Category filter**: None (all 10 categories are freshwater fish species)
- **Split**: By video/deployment ID (extracted from `dataset_<id>_frame_<n>.PNG` filename pattern, 27 unique IDs)

### salmon_computer_vision
- **Source**: [Salmon Computer Vision](https://github.com/Salmon-Computer-Vision/salmon-computer-vision) (Dropbox folder linked from the README, file `yolov6_salmon.tar.gz`)
- **Download**: Automatic (~107 GB, resumes an interrupted download). The tarball is read in one streaming pass: the YOLO labels come first in the archive, then the images, which are written directly to `data/processing/` with the dataset prefix, so the archive is never fully extracted. The archive is kept in `data/raw/` afterwards and can be deleted by hand. The `bytetrack_salmon.tar.gz` copy in the same folder (the same frames in MOT format) is not used
- **Annotations**: YOLO (normalised centre/size) -> COCO conversion, using the class list from `combined_bear-kitwanga.yaml` (the same names as the COCO files in the archive, which cover only part of the frames). 1,391,168 frames from 1,673 videos at 1920x1080, 480,058 boxes. Every frame of every video is labelled; frames without fish (962k, mostly from 7 continuous hour-long recordings) are kept. Each COCO image entry has extra fields to allow dropping consecutive frames later: `video`, `frame_index`, `video_start` (parsed from the video name; hour only for the continuous recordings) and `camera`
- **Category mapping**:
  - fish: `Coho`, `Shiner`, `Bull`, `Pikeminnow`, `Pink`, `Rainbow`, `Whitefish`, `Chinook`, `Sockeye`, `Chum`, `Jack Chinook`, `Juvenile Salmonoid`, `Steelhead`, and `Lan prey`, `Lampray` (lamprey, misspelt), `Cutthroat` (these three have no boxes)
  - non-fish: none
  - discard: none
- **Split**: By video (the frame filename without `_frame_<n>`, 1,673 videos; sorted names, `random_state`). The source's own train/valid/test split is also by video

### corefish
- **Source**: [CoReFish project page](https://warplab.github.io/corefish/) (Google Drive file `corefish_dataset.zip`)
- **Download**: Automatic, with a manual fallback (40 GB zip from Google Drive). Google Drive often refuses anonymous downloads of this file ("Quota exceeded"); the script then stops and prints how to download it signed in (in a browser, or on the server with the browser's "Copy as cURL" request) to `data/raw/corefish/corefish_dataset.zip`. The zip is kept in `data/raw/` and the frames are extracted straight from it into `data/processing/`; it can be deleted by hand afterwards
- **Annotations**: COCO format (`coco_fishonly.json`, single category `fish`, 1-indexed). 14,580 frames (162 diver transects x 90 frames at 3 fps), 100,074 boxes. Frames without fish (3,652) are kept. Image names are the source path flattened with `__`. Transect `Summer2016/JoelsShoal30mTransects061016/P6100007_0m_26s` (90 frames, 71 boxes) is annotated but its images are missing from the zip (the folder `P6100005_0m_20s` is there instead and does not match the boxes), so it is dropped. `coco_species.json` (54 species on part of the tracks, all fish) and the YOLO labels are not used
- **Category mapping**:
  - fish: `fish`
  - non-fish: none
  - discard: none
- **Split**: The authors' split (`{train,val,test}_fishonly_split.txt`), which is by survey trip: their train -> train (7,290 frames), their val + test -> val (7,200 frames). Each transect and survey trip lies in one split

## Partial Datasets

These datasets require manual download or have special dependencies.

### torsi
- **Source**: [CSIRO](https://data.csiro.au/collection/64913)
- **Download**: Automatic via the CSIRO Data Access Portal API (`instances.json` + 1,051 port-camera images, ~2.3 GB; the 849 unannotated starboard stereo images are skipped)
- **Annotations**: COCO format (already provided)
- **Category mapping**:
  - fish: `orange_roughy`, `orange_roughy_edge`, `misc_fish`, `eel`, `oreo`, `whiptail`, `chimera`, `shark`
  - non-fish: `sea_star`, `sea_urchin`, `brittle_star`
  - discard: `mollusc` (blurry, unidentifiable blobs), `cnidaria` (mostly sessile or unidentifiable), `sea_feather` (sea pens / stalked crinoids, sessile), `sea_anemone`, `coral`, `siphonophore`
- **Split**: By deployment, via the date (2019-07-13/14/15 train = 4 deployments, 2019-07-16/17 val = 2 deployments, ~19% val ratio). The 6 deployments are separate sessions in the image timestamps

### f4k
- **Source**: [UniCT OneDrive](https://bit.ly/f4k-detection-tracking)
- **Download**: Manual - download `f4k_detection_tracking.zip` and place in `data/raw/f4k/`
- **Annotations**: XML contour annotations -> bounding boxes, keyframes extracted from `gt_<video_id>.flv` videos with OpenCV
- **Category mapping**:
  - fish: `fish`
  - non-fish: none
  - discard: `open_sea`, `sea`, `rocks`, `coral`, `plant`, `dark_area`, `other`, `algae` (background regions, one polygon per scene)
- **Split**: By video ID (106-109 val, 110-124 train)

### kakadu
- **Source**: [Zenodo](https://zenodo.org/records/7250921) (`202210-KakaduFishAI-TrainingData.zip`, 5.9 GB)
- **Download**: Automatic. The zip is Deflate64-compressed, which Python's `zipfile` cannot extract, so it is extracted with the system `unzip`
- **Annotations**: COCO format (already provided; every image has at least one box)
- **Category mapping**:
  - fish: all 23 classes (`Ambassis agrammus`, `Ambassis macleayi`, `Amniataba percoides`, `Craterocephalus sturcusmuscarum`, `Denariusa bandata`, `Glossamia aprion`, `Glossogobius spp.`, `Hephaestus fuliginosus`, `Lates calcarifer`, `Leiopotherapon unicolor`, `Liza ordensis`, `Megalops cyprinoides`, `Melanotaenia nigrans`, `Melanotaenia splendida inornata`, `Mogurnda mogurnda`, `Nematalosa erebi`, `Neoarius spp.`, `Neosilurus spp.`, `Oxyeleotris lineolata`, `Scleropages jardinii`, `Strongylura krefftii`, `Sycomistes butleri`, `Toxotes chatareus`)
  - non-fish: none
  - discard: none
- **Split**: Random by image (sorted file names, `random_state`). File names are shuffled numeric ids with no deployment or video information

### marine_detect
- **Source**: [Orange marine-detect](https://github.com/Orange-OpenSource/marine-detect) (two datasets: [FishInv](https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip) + [MegaFauna](https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip))
- **Download**: Automatic
- **Annotations**: YOLO (normalised centre/size) -> COCO conversion. The two datasets are merged by filename: shared images (byte-identical) get the annotations of both
- **Category mapping**:
  - fish: `fish`, `bolbometopon_muricatum`, `chaetodontidae`, `cheilinus_undulatus`, `cromileptes_altivelis`, `haemulidae`, `lutjanidae`, `muraenidae`, `scaridae`, `serranidae`, `shark`, `ray`
  - non-fish: `turtle`, `urchin`, `sea_cucumber`, `crown_of_thorns`, `lobster`
  - discard: `giant_clam` (sessile)
- **Split**: By original split: val if the image is in `valid`/`test` of either source dataset, train otherwise
- **Special**: OzFish images are skipped to avoid duplication. EXIF orientation tags are removed (labels are relative to the stored pixels). Images without annotations are kept
