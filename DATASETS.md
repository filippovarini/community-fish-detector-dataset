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
- **Source**: VIAME DIVE collection [FishTrack23](https://viame.kitware.com/#/collection/65a140e8a4c218785d408b42); paper: Dawkins et al., "FishTrack23: An Ensemble Underwater Dataset for Multi-Object Tracking", WACV 2024
- **Download**: Automatic. The sequences in the collection's `Train` and `Test` folders are listed through the Girder API (saved to `data/raw/viame_fishtrack/sequences.json`) and each is downloaded as the zip DIVE builds on the fly (`dive_dataset/export`, media + annotations, ~53 GB in total, kept in `data/raw/viame_fishtrack/<Train|Test>/`). The export has no Content-Length and ignores Range, so a broken download restarts from scratch; finished zips are skipped on rerun. Every video appears twice in the collection; the `Video <name>.mp4` copies are unannotated duplicates and are skipped. The 3 `Sample` sequences are not used
- **Annotations**: VIAME CSV (one row per box per frame, with track ids) -> COCO. 576 sequences (474 videos, 102 image sequences) from 5 collections: SEFSC SEAMAP (430, greyscale reef-fish survey, Gulf of Mexico), Ifremer DropCam (85, baited cameras) and Game of Trawls (3, inside a pelagic trawl), Bay of Biscay, PIFSC MOUSS (13, Hawaii bottomfish) and CDFW LakeCam (45, freshwater reservoir in California). 751k boxes in 19.5k tracks over 320k frames; 155 labels. Video frames are decoded sequentially and the frame at the annotation time (`frame / fps` of the CSV header) is kept; image sequences use the annotated PNGs. All frames are saved as JPEG. Boxes are clipped to the image. Only frames with at least one fish or non-fish box are kept (~308k frames). The last annotation of a few videos is stamped at the very end of the video, where there is no frame, and is dropped. Frames are annotated at 5-30 Hz, so consecutive frames of a sequence are near-duplicates; each image keeps its `sequence`, `source_split` and `frame_index` fields so frames can be subsampled later. 21 SEAMAP image files are in two sequences (JP-23 and JRS-5, both in train) and appear twice
- **Category mapping**:
  - fish: the 148 fish species, genus and family labels listed in `FISH_LABELS` (e.g. `lutjanus_campechanus`, 204k boxes, `micropterus_salmoides`, `etelis_coruscans`, `ioglossus` (a goby genus), `carcharhinus`, `sphyrna_lewini`) and `unspecified_fish`
  - non-fish: `anomura` (10,060 boxes, squat/hermit crabs on the SEAMAP seafloor, blurry but the size of the 10-20 cm reef fish next to them), `stenocionops_spinimana` (703, a spider crab), `non_fish_animal` (332: a jellyfish in the trawl, crabs, a crayfish)
  - discard: `non_fish_bait`, `non_fish_bait_apparatus`, `non_fish_plant`
- **Split**: The authors' `Train` → train and `Test` → val (chosen by hand to cover several locations and difficulties), by SEAMAP camera drop. The authors put chunks of the same SEAMAP drop, and sometimes the same recording, in both folders (e.g. `761901310-Cam2-1..4` in Test and other `761901310` chunks in Train). SEAMAP sequences are grouped into drops: same station number (`7xxxxxxxx`, in the sequence name or image file names) or same camera system and date with start times at most 30 min apart (`SC2-Camera3-03-14-21-19-57`, in the name or image file names). A drop with any sequence in Train goes to train, which moves 27 Test sequences (25.6k boxes) to train. Result: 531 sequences → train (~283k frames, 655k fish / 11.1k non-fish boxes), 45 → val (~25k frames, 59k fish boxes, no non-fish: all non-fish labels are in Train sequences). The Ifremer `Bait-n`/`Fish-n`/`Plant-n` clips have no recording information and keep the authors' split

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

### afsc_salmon_pollock
- **Source**: [NOAA InPort item 69412](https://www.fisheries.noaa.gov/inport/item/69412) (public GCS bucket `gs://nmfs_odp_afsc/RACE/MACE/salmon_pollock_object_detection/annotated_data/`); Wilson et al. (2025), https://doi.org/10.1093/icesjms/fsaf168
- **Download**: Automatic (~17k files, ~29 GB). The bucket is listed with the GCS JSON API and every file is downloaded over HTTPS (16 threads; files already on disk with the right size are skipped, so an interrupted run resumes). `tow_data/` (full tow videos, no annotations) and `models/` are not downloaded
- **Annotations**: One custom COCO JSON per clip (184 clips named `<year>_t<tow>_vid<n>_<start>_<end>`, ~60 PNG frames each, 1280x720 or 1920x1080) -> one COCO file. Category ids differ between clips, so they are matched by name. The `file_name` fields are URLs under `.../data/<clip>/frames/` while the bucket folder is `annotated_data/`, so only the frame name is used; frames are renamed `<clip>_frame_<n>.PNG`. The CVAT frame tags (`cvat_tags`) and box attributes (track id, occluded, poor visibility, ...) are kept. The 1,095 frames without boxes (tagged "No Fish") are kept. **Frames are consecutive 30 fps video frames (2-second clips), so neighbouring frames are highly redundant**; all frames are kept
- **Category mapping**:
  - fish: `Pollock` (walleye pollock), `Salmon` (Pacific salmon), `Fish` (unidentified, fish on inspection), `Herring`
  - non-fish: none
  - discard: `Other` (80 boxes of blurry, unclear shapes, partly fish-like), together with the 76 frames containing one, so possible fish are not left unlabelled. `Shark`, `Invertebrate`, `Krill`, `Occlusion`, `No Fish`, `Fish Low`, `Fish Med`, `Fish High`, `Low Light`, `Low Visibility` are listed as categories but have no boxes
- **Split**: By tow (`<year>_t<tow>`, 7 tows; sorted, `random_state`). Val = 2019_t15 + 2019_t17 (6,540 frames, 17,588 boxes: 14,229 pollock, 3,334 salmon); train = 2019_t21, 2019_t22, 2020_t1, 2020_t5, 2020_t6 (10,382 frames, 67,198 boxes). Box counts are close to 80/20, but frame counts are ~61/39 because the val tows have fewer fish per frame

### pebl_farm_fauna
- **Source**: LILA BC ([zip](https://lilawildlife.blob.core.windows.net/lila-wildlife/community-fish-detection-dataset/pebl_farm_fauna_v1.0.zip)); PEBL CIC (2026)
- **Download**: Automatic (188 MB zip)
- **Annotations**: COCO Camera Traps JSON (boxes already `[x, y, w, h]` in pixels, string image ids). 1,035 frames (1920x1080 and 1080x1920) from 57 videos at 5 anonymised cameras on 2 UK seaweed/shellfish farms, 2,464 boxes; every frame has at least one box. Frame names repeat across videos (`images/<farm>/<video>/frame_<n>.jpg`), so images are flattened to `<video>_frame_<n>.jpg` and given integer ids; the `location`, `seq_id`, `frame_num` and `datetime` (video start) fields are kept on each image. Boxes were drawn by people or proposed by detectors (PEBL RF-DETR and the CFD) and verified by a person (`source` field); a few animals neither detector proposed may be unboxed
- **Category mapping**:
  - fish: `Fish`, `Scooter` (dragonets and gobies), `Flatfish`, `Elasmobranch` (no boxes in v1.0)
  - non-fish: `Hermit crab`, `Crab`, `Jellyfish` (moon and lion's mane jellies, large), `Starfish`, `Gastropod` (whelk-like sea snails), `Squid` and `Shrimp` (no boxes in v1.0)
  - discard: none
- **Split**: By camera location (the image `location` field): `farm_A_sc14` (151 frames) and `farm_A_bed1` (39) → val (190 frames, 293 fish / 249 non-fish boxes); `farm_A_sc11` (780), `farm_A_cam` (33, 2020) and `farm_B_cam` (32) → train (845 frames, 1,014 fish / 908 non-fish boxes). 133 of the 143 jellyfish boxes are at sc14, so they end up in val

### corefish
- **Source**: [CoReFish project page](https://warplab.github.io/corefish/) (Google Drive file `corefish_dataset.zip`)
- **Download**: Automatic, with a manual fallback (40 GB zip from Google Drive). Google Drive often refuses anonymous downloads of this file ("Quota exceeded"); the script then stops and prints how to download it signed in (in a browser, or on the server with the browser's "Copy as cURL" request) to `data/raw/corefish/corefish_dataset.zip`. The zip is kept in `data/raw/` and the frames are extracted straight from it into `data/processing/`; it can be deleted by hand afterwards
- **Annotations**: COCO format (`coco_fishonly.json`, single category `fish`, 1-indexed). 14,580 frames (162 diver transects x 90 frames at 3 fps), 100,074 boxes. Frames without fish (3,652) are kept. Image names are the source path flattened with `__`. Transect `Summer2016/JoelsShoal30mTransects061016/P6100007_0m_26s` (90 frames, 71 boxes) is annotated but its images are missing from the zip (the folder `P6100005_0m_20s` is there instead and does not match the boxes), so it is dropped. `coco_species.json` (54 species on part of the tracks, all fish) and the YOLO labels are not used
- **Category mapping**:
  - fish: `fish`
  - non-fish: none
  - discard: none
- **Split**: The authors' split (`{train,val,test}_fishonly_split.txt`), which is by survey trip: their train -> train (7,290 frames), their val + test -> val (7,200 frames). Each transect and survey trip lies in one split

### fjordfish
- **Source**: [Zenodo](https://zenodo.org/records/17950781)
- **Download**: Automatic (`FjordFish.zip`, 6.1 GB)
- **Annotations**: YOLO TXT -> COCO conversion (28 classes in `data.yaml`; source train/val folders merged, labels matched to images by file stem because `2515.jpg` is in `images/train` but its label is in `labels/val`). Mixed resolutions: BRUV and mesocosm video frames plus diver DSLR photos
- **Category mapping**:
  - fish: `ballan`, `cuckoo_female`, `cuckoo_male`, `corkwing_female`, `corkwing_male`, `goldsinny`, `rock_cook`, `labrids_unknown`, `cod`, `cod_yoy`, `haddock`, `pollack`, `poor_cod`, `saithe`, `whiting`, `gadines_unknown`, `spiny_dogfish`, `sharks_unknown` (no boxes), `ling`, `two_spot_goby`, `black_goby`, `sand_goby`
  - non-fish: `brown_crab`, `green_crab`, `hermit` (hermit crabs), `lobster`, `decapods_unknown` (all crabs on inspection)
  - discard: `unknown` (no boxes)
- **Split**: By blocks of 100 consecutive image ids (`<id> // 100`, 31 blocks; sorted, `random_state`). Images are named `1.jpg`...`3002.jpg` in source order, so neighbouring ids are frames of the same video or photos of the same series; no video/site id is published. The authors' train/val folders are a random per-image split and are ignored

### n_marine
- **Source**: [Government of Canada Open Data Portal](https://open.canada.ca/data/en/dataset/2ae46860-f82a-4127-bb1f-b02e36ef6a70)
- **Download**: Automatic (15 PNG zips, ~23 GB, plus `nmarine_csv.zip`; resumable). The zips are kept in `data/raw/` and never extracted there: images are read from them straight into `data/processing/` as `<video>_<frame>.png`. The server's firewall rejects range requests near the end of a zip, so a resume that fails there needs the `.part` file deleted
- **Annotations**: One CSV per video (`img`, `TL_x/TL_y/BR_x/BR_y` in pixels, `marine_class`, `species`) -> COCO conversion. Boxes are clipped to the 1920x1080 frame (some extend up to ~25 px outside). 20,936 annotated frames + 3,000 background frames (frames in the zips with no CSV row), which are kept as images without annotations
- **Category mapping**:
  - fish: `Cod` (Gadus morhua), `Roughhead_grenadier` (Macrourus berglax), `Atlantic_halibut` (Hippoglossus hippoglossus), `Redfish_mentella` (Sebastes mentella), `Thorny_skate` (Amblyraja radiata), `Striped_wolffish` (Anarhichas lupus), `Spinytail_skate` (Bathyraja spinicauda), `Wolffish` (Anarhichas denticulatus)
  - non-fish: `Whelk` (Buccinum undatum, a crawling sea snail; 869 boxes, all from one video)
  - discard: none
- **Split**: By camera deployment. Videos are named `<camera>_<YYYYMMDD>_<HHMMSS>` and consecutive 5-minute videos of one deployment film the same pot. 5 deployments: camera 097 on 16-17 Jul (one deployment past midnight), 097 on 19 Jul, 116 on 21 Jul, 130 on 19 Jul, 136 on 19 Jul. Camera 097 on 19 Jul (3,494 annotated frames, ~17%) -> val, the rest -> train

### wio_reef_fish
- **Source**: [Zenodo](https://zenodo.org/records/21360120)
- **Download**: Automatic (`WIO-ReefFish_v1.0.0.zip`, 636 MB, resumes an interrupted download, MD5 checked)
- **Annotations**: YOLO TXT -> COCO conversion (source train/valid/test merged). 1,000 frames at 3840x2160 from 50 diver-operated video transects in Kenya (2023) and Tanzania (2025), 6,768 boxes. Class ids follow `data.yaml`; the per-class box counts are checked against `metadata/annotations_summary.csv`. Only fish of the 24 families are boxed, so fish of other families may be unlabelled
- **Category mapping**:
  - fish: all 24 classes (`Acanthuridae`, `Anthiadidae`, `Aulostomidae`, `Balistidae`, `Caesionidae`, `Chaetodontidae`, `Epinephelidae`, `Haemulidae`, `Holocentridae`, `Kyphosidae`, `Labridae`, `Lethrinidae`, `Lutjanidae`, `Microdesmidae`, `Monacanthidae`, `Mullidae`, `Pempheridae`, `Pomacanthidae`, `Pomacentridae`, `Priacanthidae`, `Scarinae`, `Siganidae`, `Tetraodontidae`, `Zanclidae`)
  - non-fish: none
  - discard: none
- **Split**: By location. The video id (filename before `_frame_<n>`) gives the location through the authors' `alternative_splits/transect/VIDEO_ASSIGNMENT.csv` (47 videos, 10 locations). Diani (Kenya) and Mikindani (Tanzania) go to val (223 images, 1,341 boxes); the other 8 locations go to train (777 images, 5,427 boxes). Videos `GX011330`, `GX011332`, `GX011334` (12 images) are missing from the CSV; by GoPro file number they were filmed between the Kunduchi and Kilwa videos, so they go to train and neither location is in val. The source's headline split is random by frame and is not used

### obsea
- **Source**: [Zenodo](https://doi.org/10.5281/zenodo.14888328) (`obsea_dataset_v4.1.zip`)
- **Download**: Automatic (3.8 GB, resumes an interrupted download, checked against the Zenodo MD5)
- **Annotations**: YOLO TXT -> COCO conversion. 4,120 images, 35,840 boxes in 74 classes. The YOLO class ids follow the order of the class names in `obsea_dataset.json`; the box count per class listed there is checked against the labels. Images with no boxes left after mapping (413) are kept
- **Category mapping**:
  - fish: all 63 fish species/variant classes (variants: `(back)` = fish far in the background, `(night)`, `(alev.)` = juveniles, `(m)`/`(f)`), plus `unknown` and `unknown sp` (48 crops of each checked: nearly all blurry or distant fish; a few boxes in `unknown sp` are unclear)
  - non-fish: `Cranc` (crab), `Octopus vulgaris`, `Rhizostoma pulmo` and `Chrysaora hysoscella` (large jellyfish), `Asteroidea` (starfish)
  - discard: `Diver`, `Delphinus (dead)` (a dolphin carcass on the seabed), `Posidonia oceanica (shot)` (seagrass), `Gateropodae` (small snail shells)
- **Split**: By camera, from the filename. Cameras `AIPC608UW_10_167` and `C4k0193` -> val (834 images, 20%); the other cameras and the videos/images without a camera id -> train

### digi4eco
- **Source**: [Zenodo](https://doi.org/10.5281/zenodo.20701225) (`DIGI4ECO_tracking_dataset.zip`)
- **Download**: Automatic (139 MB, resumes an interrupted download, checked against the Zenodo MD5)
- **Annotations**: CVAT XML for video 1.1 (multi-object tracks over 4 MP4 videos) -> COCO conversion. Every annotated frame is extracted from the videos with OpenCV, and every box with `outside="0"` and `occluded="0"` becomes a COCO box; track ids are dropped. The visible box count of each XML is checked against its `annotation_stats`. Boxes with `occluded="1"` (11,798, 9.4% of the visible boxes, ~95% of them interpolated by CVAT) are dropped: in 96 random crops (24 per sequence) most showed no fish, only the pillar or rope the fish was behind, open water, or a closer fish hiding it; only about 1 in 5 showed part of the fish. Ancona, SmartBay1 and SmartBay2 decode to exactly their annotated frame count, so CVAT frame k is the k-th decoded frame. The OBSEA video has 148 frames for 150 annotated ones (timestamps on a 1/30 s grid with two gaps), so its frames are matched by timestamp and CVAT frames 1 and 148 (15 non-occluded boxes) are skipped. 5,849 images, 114,207 boxes; every frame keeps at least one box. Each COCO image entry has extra fields to allow dropping consecutive frames later: `video`, `frame_index` and `site`
- **Category mapping**:
  - fish: `Fish`, `Apogon sp` (2 tracks in SmartBay1, 62 boxes; cardinalfish, crops checked)
  - non-fish: none
  - discard: none
- **Split**: By site, from the sequence name in the filename. Ancona (Adriatic Sea) -> val (1,200 images, 22,441 boxes, 20%); OBSEA and SmartBay1/SmartBay2 (Galway Bay, two cameras) -> train (4,649 images, 91,766 boxes). The OBSEA clip (camera ANERIS-EMUAS1, July 2025) is not from the cameras that `obsea` puts in val

### deepecomar
- **Source**: [Zenodo](https://zenodo.org/records/7534425) (Labelled Mediterranean Fish Dataset, DEEPECOMAR project)
- **Download**: Automatic (`DEEPECOMAR_DATASET_YOLO.zip`, 638 MB, resumes an interrupted download)
- **Annotations**: YOLO TXT (Roboflow YOLOv5 export, 25 classes in `data.yaml`) -> COCO conversion, source train/valid merged. Roboflow renames files to `<name>_jpg.rf.<hash>.jpg`; the suffix is stripped. 74 frames appear twice in train as identical copies (same pixels and labels), so one image per original name is kept: 1,590 files -> 1,516 images. Three sources, told apart by filename: `vivo_<YYYYMMDD>_<HHMMSS>M` (fixed time-lapse camera on an underwater structure, 886 images, 2018-19), `vlcsnap-<YYYY-MM-DD>-...` (video snapshots on 4 dates in 2021, 467 images), `E01L_-<n>-_frame_<n>` (one stereo video, left camera, 163 frames)
- **Category mapping**:
  - fish: all 22 named taxa (`Chromis chromis`, `Coris julis`, `Dentex dentex`, `Diplodus annularis`, `Diplodus puntazzo`, `Diplodus sargus`, `Diplodus sp-`, `Diplodus vulgaris`, `Epinephelus marginatus`, `Lithognathus mormyrus`, `Mugilidae prob Chelon`, `Mullus sp-`, `Oblada melanura`, `Pomatous salator`, `Sarpa salpa`, `Seriola dumerilii`, `Serranus cabrilla`, `Serranus scriba`, `Sparus aurata`, `Sphyraena sp-`, `Symphodus sp-`, `labrid unid-`), `Afish poly` (unidentified fish, crops checked), `small` (fish < ~100 px^2 per the authors; boxes are 1-20 px, kept so these fish are not learned as background)
  - non-fish: none
  - discard: `taca` (Spanish for stain: lens smudges, crops checked)
- **Split**: By location. Val = video snapshots from 2021-04-29 and 2021-05-03 (separate dives; 296 images, 810 boxes, ~20%); train = the fixed camera, the other two video dates and the stereo video (1,220 images, 16,347 boxes). The authors' train/valid folders are a random per-image split and are ignored

### seatizen_atlas
- **Source**: [Zenodo, Seatizen Atlas image dataset v1.1.0](https://zenodo.org/records/13951614) (paper: https://doi.org/10.1038/s41597-024-04267-z). The latest version of the record ([20459523](https://zenodo.org/records/20459523)) has only image-level multilabel CSVs, so v1.1.0 is used
- **Download**: Automatic. `20241018_segmentation_annotations.json` (351 MB, resumable) is downloaded; `images.zip` (35 GB, 14,493 images) is not: only the 1,200 segmented images (~0.6 GB) are read from it with HTTP Range requests into `data/raw/seatizen_atlas/images/`
- **Annotations**: COCO instance segmentation (polygons, 0-indexed image/annotation ids, category ids 482-1020) -> COCO boxes only (the source `bbox` of each polygon), ids shifted to 1-indexed. 1,200 images, 34,709 instances in 37 classes. Images with no fish or non-fish (304) are kept as background. No EXIF-rotated images (checked in the script). `20210218_REU-Hermitage_scuba_4_00563.jpg` is dropped: it is 1920x1080 in the zip but annotated as 3840x2160, and its one kept box (a sea cucumber), scaled or not, frames no animal
- **Category mapping**:
  - fish: `Fish` (8,494)
  - non-fish: `Sea_cucumber` (1,186), `Sea_urchin` (27), `Turtle` (7), `Other_starfish` (5)
  - discard: `Giant_clam`, `Sea_anemone`, `Sponges` (sessile); `Homo_Sapiens`, `Human_object`, `Waste`, `Trample`, `Useless`; all coral (`Acropora_*`, `Non-acropora_*`, `Living_coral`, `Dead_coral`, `Bleached_coral`, `Soft_coral`, `Gorgonian`), algae (`Algal_Assemblage`, `Algae_*`), seagrass (`Syringodium_isoetifolium`) and substrate (`Sand`, `Rubble`, `Rock`) classes
- **Split**: By site, from the second token of the file name (`<date>_<site>_<vehicle>_...`). Mayotte (`MYT`) -> val (322 images, 3,356 fish, 33 non-fish); Reunion (`REU-Hermitage`), Europa Island (`ATF-Europa`) and Mauritius (`MUS-*`) -> train (877 images, 5,138 fish, 1,191 non-fish). Frames from the same session are often consecutive video frames, so many are near-duplicates

### yolo_bruv
- **Source**: [Zenodo](https://doi.org/10.5281/zenodo.19594599) (`YOLO-BRUV_data.zip`)
- **Download**: Automatic (415 MB, resumes an interrupted download, checked against the Zenodo MD5); only `YOLO-BRUV_data/data/Annotations/` is extracted
- **Annotations**: EventMeasure point exports (`EMOutputs/Raw/G1-G6_Points.txt`, tab-separated, one row per animal: `ImageCol`/`ImageRow` top-left corner and `RectWidth`/`RectHeight` in pixels, `Genus`, `Species`) -> COCO conversion. The raw exports are used, not `EMOutputs/Corrected/`, which only regroups the species into the authors' training classes. The 2,721 JPEGs (1920x1080) are 2,708 frames exported in 6 annotation groups:
  - 68 frames of G1 in which animals are marked with points (no box; other animals in them are not marked) are dropped
  - 12 frames were exported into two or three groups and annotated separately (e.g. 3 hermit crabs in one copy, none in the other): the copy with the most boxes is kept (13 copies dropped)
  - 455 frames carry only the empty-frame marker and are kept as background
  - no box reaches past the image; boxes 1 pixel or less wide or high would be dropped (none)
  - 2,640 frames, 15,919 boxes before mapping. Each COCO image keeps `video`, `chapter`, `frame` and `group` (from the file name, e.g. `01_RGX030003.MP4_0009008.jpg` is chapter 03 of video 0003, frame 9008)
- **Category mapping** (source category is `Genus species`, or the species alone when the genus is missing):
  - fish: `Limanda-Platichthys limanda-flesus` (dab or flounder, 7,861), `Melanogrammus aeglefinus` (haddock, 3,469), `Pleuronectiformes spp` (1,472), `minutus` (gobies, 92), `Myoxocephalus scorpius` (44), `Eutrigla gurnardus` (31), `Scyliorhinus canicula` (30), `Pleuronectes platessa` (28), `Callionymus lyra` (18), `Gadus morhua` (12), `Merlangius merlangus` (4), `Molva molva` (4), `Dipturus intermedius` (3), `Limanda limanda` (1), `Trisopterus minutus` (1)
  - non-fish: `Pagurus bernhardus` (hermit crab, 2,003), `Asterias rubens` (starfish, 637), `Buccinum undatum` (common whelk, 18; one or two live whelks crawling near the bait in video 0003, crops checked), `Cancer pagurus` (11), `Ophiura albida` (brittle star, 4, crops checked), `vulgaris` (1, genus missing; a squid, crop checked)
  - discard: `Botryllus spp` (175 boxes, plus points), a marker placed in open water to export empty frames (not the sea squirt; crops checked)
- **Split**: By video. Every frame comes from one of 19 GoPro recordings (camera deployments), and frames of one recording can be a few seconds apart. Whole videos are shuffled (`random_state`) and moved to val until it holds 20% of the frames: val videos 0006, 0007, 0008, 0011, 0021, 0024, 0025, 0027 (574 frames, 21.7%; 4,111 fish / 177 non-fish boxes), train the other 11 (2,066 frames; 8,959 / 2,497). The footage covers two of the four study sites, so val is not from a site unseen in train

### griffith_seagrass
- **Source**: [PANGAEA 926930](https://doi.pangaea.de/10.1594/PANGAEA.926930) (paper: https://doi.org/10.3389/fmars.2021.629485)
- **Download**: Automatic (`Fish_automated_identification_and_counting.zip`, 3.7 GB, resumes an interrupted download). The outer zip holds the annotations, `dataset_images.csv` and an inner `luderick-seagrass.zip` with the 4,280 frames (flat, 1920x1080), which is extracted to `data/raw/griffith_seagrass/Fish_automated_identification_and_counting/images/`
- **Annotations**: COCO JSON (`luderick_seagrass_all.json`, boxes and polygons, image/annotation/category ids 0-indexed) -> ids shifted to 1-indexed. 25 boxes that stick out of the frame by 1-2 px are clipped. Every frame has at least one box. Frames were extracted from video at 5 fps, so consecutive frames are often near-duplicates
- **Category mapping**:
  - fish: `luderick` (Girella tricuspidata, 9,304), `bream` (Acanthopagrus australis, 125)
  - non-fish: none
  - discard: none
- **Split**: By estuary, from the `dataset` column of `dataset_images.csv`. Val = Tallebudgera Creek (the authors' "novel test" site, deployments 16-18; 764 images, 1,066 boxes); train = Jack Evans Boat Harbour (the authors' "Jack Evans A" train and "Jack Evans B" test subsets, deployments 1-15; 3,516 images, 8,363 boxes). A and B are not used as separate splits because they share deployments and one video (`2205_Luderick_25`)

### med_fish_census
- **Source**: [Dryad, doi:10.5061/dryad.f7m0cfz6f](https://datadryad.org/dataset/doi:10.5061/dryad.f7m0cfz6f) (paper: https://doi.org/10.1016/j.ecoinf.2024.102959)
- **Download**: Automatic with Dryad API credentials (`DRYAD_CLIENT_ID` + `DRYAD_CLIENT_SECRET`, or `DRYAD_TOKEN`, in the environment or `.env`; create them under My account > API on datadryad.org). The token resolves the download to a signed S3 URL and only `Data_T` (4.4 GB, 39.6k files) plus the class lists are read from `autoUC_Data.zip` (121.7 GB) with HTTP Range requests (`HttpRangeFile`). Without credentials, download the zip manually to `data/raw/med_fish_census/autoUC_Data.zip`; the script picks it up and extracts `Data_T` from it. `datadryad.org/downloads/file_stream/...` sits behind a browser bot check and cannot be scripted
- **Annotations**: YOLO TXT -> COCO. `Data_T/{trainSet,testSet}/images` with two label sets of identical fish boxes: `labels_allSpecies` (46 classes, used) and `labels_Reduced` (19 species + Other; used for the 118 train images without a `labels_allSpecies` file). 387 `labels_allSpecies` files without an image are ignored. 63 frames (videos GH010143_1, GH010149, GH010211_2) are byte-identical in both splits with the same labels and are kept once. Frames are `<video>_<frame>.jpg` (12,421 at 1920x1080, 609 at 2704x1520, 14 at 1280x960). Frames without kept boxes (2,233, mostly background/diver-only) are kept as background. 6 fish boxes with zero width or height in the source are dropped
- **Category mapping**:
  - fish: the 40 species classes (`Chromis_chromis` ... `Diplodus_anularis`, see `FISH_SPECIES` in the script) and `Other` (rare species grouped)
  - non-fish: `jellyfish` (127, Pelagia noctiluca, crops checked), `seaurchin` (78)
  - discard: `background` (6,685; hard negatives such as rocks, seagrass, wreck parts, crops checked), `diver` (2,061), `drone` (no boxes), `fish` (4 boxes, all seagrass leaves)
- **Split**: The authors' split by video (testSet -> val, trainSet -> train), except that GoPro recordings with clips in both of their splits (GH010003, GH010004, GH010049, GH010144, e.g. GH010003 in test and GH010003_2 in train) go entirely to val. No site ids are in the data. Val: 3,994 images, 28,867 fish, 5 non-fish (~30%); train: 9,050 images, 39,723 fish, 196 non-fish

### eyesea
- **Source**: [PNNL DataHub, EyeSea Underwater Video](https://data.pnnl.gov/group/nodes/dataset/12978); Xu & Matzner (2018), [doi:10.1109/CSCI46756.2018.00067](https://doi.org/10.1109/CSCI46756.2018.00067)
- **Download**: Automatic (87 GB tar from the PNNL Globus endpoint, resumes an interrupted download). The tar holds a second tar, `eyesea.tar`, which is streamed without being written to disk; only `train_x/y` and `{voith,wells,orpc}_test_x/y` are extracted. `videos/` (source videos) and `keras-yolo3-master/` (the authors' model and debug output) are skipped. The outer tar is deleted after extraction
- **Annotations**: Pascal VOC XML, one file per PNG frame in the matching `_y` folder -> COCO conversion. Image size is read from the PNG, and boxes are clipped to the image. Frames whose label file has no boxes are kept as empty images; frames without a label file are skipped. Frames come from video at 10-30 fps, so consecutive frames are very similar. Voith Hydro frames are a 2x2 mosaic of the turbine's camera views (one panel often "NO VIDEO"). 70,195 frames (all of them labelled; one Wells label file has no image), 45,652 boxes; 48,125 frames have no fish. Per site, the frames with fish match the paper's Table I (Voith 7,663, Wells 13,405, Igiugig 1,002). Two clips have descriptive names instead of the usual prefix and are assigned by name: `voith_Fish_with_operational_turbine_2` (450 frames) and `orpc_adults_then_smolt` (280 frames)
- **Category mapping**:
  - fish: `fish` (the only class)
  - non-fish: none
  - discard: none
- **Split**: By site, from the filename prefix: Voith Hydro (`Voith_*`/`voith_*`, tidal turbine, Scotland, grayscale) -> val (15,918 frames, 20,914 boxes, 23%), from both the authors' train and test folders; Wells Dam (`WellsDam_*`, 24,000 frames, 23,496 boxes) and Igiugig/ORPC (`<YYYYMMDD>_<HHMMSS>*`/`orpc_*`, 30,277 frames, 1,242 boxes) -> train (54,277 frames, 24,738 boxes). The authors' split (about 80/20 per site) is ignored because, for Voith and Wells, it puts later 30 s segments of the same recording in test

### seaclear
- **Source**: [4TU.ResearchData, Seaclear Marine Debris Detection & Segmentation Dataset](https://doi.org/10.4121/4f1dff25-e157-4399-a5d4-478055461689) (paper: https://doi.org/10.1038/s41597-024-03759-2)
- **Download**: Automatic from the [Kaggle mirror](https://www.kaggle.com/datasets/jocelyndumlao/seaclear-marine-debris-detection-and-segmentation) via kagglehub (1.6 GB zip, no login). 4TU serves a single .rar and refused connections when the script was written
- **Annotations**: One COCO file (`dataset.json`, instance polygons, 0-indexed image/annotation ids, 1-indexed categories) -> COCO boxes only (the source `bbox`), ids renumbered from 1. Images sit in `<site>/<camera>/` folders while COCO file names have no folder; names are unique across folders, so the site is recovered by file name. 8,610 images, 31,555 instances in 40 classes. Marseille (3,441 images) has no animal labels; a sample showed no unlabelled fish, so those images are kept as background. Fish are labelled in only 81 images
- **Category mapping**:
  - fish: `animal_fish` (985)
  - non-fish: `animal_urchin` (6,462), `animal_etc` (3,749; ~50 crops from all sites were sea cucumbers, with a few unclear blobs at Jakljan), `animal_starfish` (17)
  - discard: `animal_shells` (empty shells, oysters, snails), `animal_sponge` (sessile), `plant`, `unknown_instance` (crops are debris and rubble), the 32 debris classes (`can_metal`, `bottle_glass`, `tire_rubber`, ...) and robot parts (`rov_cable`, `rov_tortuga`, `rov_vehicle_leg`, `rov_bluerov`)
- **Split**: By site. 950 of the 985 fish are at Lokrum, so Lokrum stays in train. Val = Jakljan + Slano (545 images, 26 fish, 77 non-fish); train = Bistrina, Lokrum, Marseille (8,065 images, 959 fish, 10,151 non-fish). Many images are frames from the same ROV video, so neighbours are near-duplicates

### roboflow_aquarium
- **Source**: [Roboflow Universe, Aquarium Combined](https://universe.roboflow.com/brad-dwyer/aquarium-combined) (also https://public.roboflow.com/object-detection/aquarium)
- **Download**: Automatic, via `kagglehub` from the Kaggle mirror [slavkoprytula/aquarium-data-cots](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots) (67 MB, no login). Roboflow Universe blocks scripted access (403) and its API needs a key
- **Annotations**: YOLO TXT (Roboflow YOLOv5 export, 7 classes in `data.yaml`) -> COCO conversion, source train/valid/test merged (638 images, no duplicates). Roboflow renames files to `<name>_jpeg_jpg.rf.<hash>.jpg`; the suffix is stripped. Photos taken through tank glass at two US aquariums, resized to fit 1024x1024. Two cameras, told apart by filename: `IMG_2xxx`-`IMG_3xxx` (437 images, the only ones with penguins and starfish) and `IMG_83xx`-`IMG_85xx` (201 images, stills and `MOV-<n>` video frames)
- **Category mapping**:
  - fish: `fish`, `shark`, `stingray`
  - non-fish: `jellyfish` (large sea nettles and spotted jellies), `starfish`
  - discard: `penguin`, `puffin` (birds, almost always photographed on rocks out of the water, crops checked). The 136 images that contain only birds are kept without boxes
- **Split**: By camera. Train = `IMG_2xxx`-`IMG_3xxx` (437 images, 1,842 fish + 433 non-fish boxes); val = `IMG_83xx`-`IMG_85xx` (201 images, 1,365 fish + 377 non-fish boxes, ~32%; no starfish). The provided train/valid/test split is random per image and is ignored

### ruod_r
- **Source**: annotations [RUOD-R](https://github.com/RSSL-MTU/RUOD-R) (`Annotations/RUOD-R/COCO`, commit `988939f`), images [RUOD](https://github.com/xiaoDetection/RUOD) (release `RUOD.tar.partaa` + `RUOD.tar.partab`)
- **Download**: Automatic (annotations 58 MB, images 3.7 GB in two tar parts; resumes interrupted downloads, every file MD5 checked; the parts are streamed into `tar` and only `RUOD/RUOD_pic` is extracted)
- **Annotations**: COCO (both RUOD-R splits merged; image ids repeat across the splits, so images and boxes are renumbered). The full RUOD-R version is used, not "Filtered RUOD-R" (which drops the 302 photos with more than 100 boxes and about 58,000 fish boxes). 756 groups of byte-identical photos (1,643 photos; in 322 groups the copies are on both sides of the RUOD split, and the copies were annotated separately, with the same box count in only 269 groups): only the copy with the most boxes is kept (887 copies dropped). Photo `008431.jpg` is dropped: it is stored rotated (EXIF orientation 6) and none of its 5 boxes fits any orientation. Boxes reaching past the image edge are clipped (2,207 in all 14,000 photos); 28 boxes 1 pixel or less wide or high are dropped. 13,112 images, 253,530 boxes before mapping; images left without boxes (1,350) are kept. Each image keeps `ruod_split` and `source_group`. The 300 `Environment_pic` photos of the archive are copies of RUOD photos and are not used
- **Category mapping**:
  - fish: `fish`, except 89 boxes on bottlenose dolphins (4 diving videos and 2 single photos), which are relabelled `dolphin` -> non-fish. They are listed in `datasets/ruod_r_dolphins.json` and were found by checking every fish box at least 10% of the image in the videos with dolphins, and every low-saturation (grey) fish box at least 15% of the image in the whole dataset (679 boxes: sharks, rays, remoras, groupers, bass, no other dolphins). Smaller, distant dolphins may remain labelled fish
  - non-fish: `echinus` (sea urchin), `holothurian` (sea cucumber), `starfish`, `cuttlefish`, `turtle`, `jellyfish`, and `dolphin` (above). Many echinus, holothurian and starfish boxes in murky green seabed frames are faint blobs
  - discard: `scallop` (a shell; 29,153 boxes), `corals`, `diver`
- **Split**: By source group, not the RUOD train/test split. Photos are numbered in blocks from the same video (e.g. 827 consecutive 4K frames), and the RUOD split assigns them randomly, so 3,434 neighbouring frames straddle train and test. Photos are grouped (union-find) when they are consecutive file numbers of the same size, near-identical (normalised 32x18 grayscale thumbnails, correlation >= 0.95 at the same size or >= 0.98 at different sizes) or byte-identical: 5,145 groups, the largest 1,707 photos. Whole groups are shuffled (`random_state`) and moved to val until it holds 20% of the photos: val 2,624 images (1,103 groups, 26,836 fish / 10,461 non-fish boxes), train 10,488 images (4,041 groups, 77,366 / 57,476). Remaining overlap: 78 val photos have a train photo with correlation 0.95-0.98 at a different size, about two thirds of them the same scene rescaled, cropped or colour-corrected; a lower threshold chains thousands of murky seabed frames into one group of a quarter of the photos

### uiis10k
- **Source**: [UIIS10K](https://github.com/LiamLian0727/UIIS10K), data on [Hugging Face](https://huggingface.co/datasets/LiamLian0727/UIIS10K) (`UIIS10K.zip`, revision `5c43163`)
- **Download**: Automatic (394 MB, resumes an interrupted download, MD5 checked)
- **Annotations**: COCO JSON with instance polygons (`multiclass_train.json`, `multiclass_test.json`; 10,048 photos, 41,862 instances). Boxes are recomputed from the polygons (9 source boxes do not match their polygon, 2 are `[0, 0, 0, 0]`) and clipped; 4 boxes 1 pixel or less wide or high are dropped. 39 fish and non-fish instances whose polygon joins separate animals (parts further apart than half the larger part) get one box per group of parts; 24 boxes with IoU >= 0.85 to a box of the same output class are dropped. Photos:
  - 359 photos that are also RUOD photos (some colour-corrected) are dropped, listed in `datasets/uiis10k_in_ruod.json` (high-pass thumbnail correlation >= 0.81 with a RUOD photo, every match checked; below 0.81 the matches are different dark frames). `ruod_r` annotates them more densely
  - copies of the same photo (byte-identical, or normalised high-pass 64x48 grayscale thumbnails with correlation >= 0.96; mostly one photo at 640x480 and 320x240, or recoloured) were annotated separately and sometimes differently: of each of the 188 groups only the copy with the most fish and non-fish boxes is kept (191 copies dropped; 74 groups had copies in both UIIS10K splits)
  - 9,498 photos kept; 3,285 of them have no fish or non-fish box (corals, divers, wrecks or trash only) and are kept. Each COCO image keeps its `uiis_split`
- **Relabelling**: the UIIS10K classes do not follow ours ("fish" is "fish and other marine vertebrates"; "mollusk" holds starfish, sea urchins and octopus but also clams, anemones, corals and rays; "reptiles" and "arthropoda" hold eels and other fish), so `datasets/uiis10k_relabels.json` gives a new label to 2,928 boxes. Every fish, mollusk, reptiles and arthropoda box (24,451) was first labelled by Claude Haiku 5.5 from a crop with the box drawn (Message Batches API, 8 crops per request, 20 labels, $0.48), then checked by eye: the 1,112 fish boxes Haiku did not call a fish or unclear (177 of them in the full photo), and every mollusk (2,333), reptiles (366) and arthropoda (388) box, with its mask outline. The 5,731 fish boxes Haiku called unclear (mostly small fish) were not checked. Boxes that could not be identified are labelled `unclear` and discarded. Changes over all 10,048 photos:
  - fish (21,364): 122 marine mammal (dolphins, whales, belugas, orcas, seals, sea lions, walruses, manatees, polar bears), 44 coral or sponge, 22 unclear, 18 octopus/squid/cuttlefish, 16 human, 12 bird (penguins), 6 nudibranch, 4 shell, 3 anemone, 3 turtle, 3 other reptile (crocodiles, marine iguana), 2 jellyfish, 2 object, 2 plant, 2 sea cucumber, 1 crustacean, 1 starfish
  - mollusk (2,333, all relabelled): 646 unclear, 306 starfish (feather stars too), 292 anemone, 229 sea urchin (sand dollars too), 159 coral or sponge, 155 shell (clams, giant clams, snails, empty shells), 139 octopus/squid/cuttlefish, 136 nudibranch, 106 jellyfish, 74 fish (rays, clownfish, trumpetfish), 40 sea cucumber, 25 plant, 13 object, 7 crustacean, 4 turtle, 1 human, 1 marine mammal
  - reptiles (366): 38 unclear, 29 fish (moray eels, scorpionfish, flatfish), 10 marine mammal, 7 bird, 6 sea cucumber, 3 octopus/squid/cuttlefish, 1 crustacean, 1 nudibranch, 1 shell; 270 kept (sea turtles, sea snakes)
  - arthropoda (388): 161 unclear, 24 fish (lionfish, seadragons, pipefish), 12 octopus/squid/cuttlefish, 11 crustacean, 9 shell, 6 coral or sponge, 4 anemone, 4 sea urchin, 2 nudibranch, 2 sea cucumber, 1 turtle; 152 kept (crabs, shrimp, lobsters)
- **Category mapping** (after relabelling):
  - fish: `fish`
  - non-fish: `reptiles`, `arthropoda`, `marine mammal`, `bird`, `turtle or sea snake`, `other reptile`, `octopus, squid or cuttlefish`, `crustacean`, `starfish`, `sea urchin`, `sea cucumber`, `jellyfish`
  - discard: `corals`, `plants`, `ruins`, `garbage`, `robots`, `human`, `mollusk` (no box left), and the relabelled `human`, `shell`, `anemone`, `coral or sponge`, `nudibranch` (size doubtful), `plant`, `object`, `unclear`
- **Split**: The UIIS10K split (train -> train, test -> val). The photos have no location, video or source information and the file names are shuffled (consecutive photos are unrelated), so the random official split is kept, after dropping copies. Photos of the same shot that are not copies (thumbnail correlation 0.90-0.96, e.g. two frames of one deep-sea video with a different object in view) go to the split of the one with the most boxes (3 photos moved from train to val): val 1,907 images (3,845 fish / 268 non-fish boxes, 20%), train 7,591 (16,092 / 1,131). No val photo has a train photo with correlation >= 0.90 or a byte-identical copy

### kso_mussel_bruv
- **Source**: [Zenodo 14713926](https://doi.org/10.5281/zenodo.14713926) (`coolblue_LTA_model.zip`, folder `model_dataset`)
- **Download**: Automatic (415 MB, resumes an interrupted download, checked against the Zenodo MD5); only `model_dataset/{train,valid,test}/{images,labels}` is extracted (not the YOLOv8 weights or the result videos)
- **Annotations**: YOLO (Roboflow export, 6 classes; one label is a polygon, converted to its box) -> COCO, the three Roboflow splits merged. 910 frames (640x360) named `<farm|control>_<day>_<month>_frame_<n>`, from 8 recordings (farm and control on 6 Mar, 12 Apr, 17 Apr and 4 May 2024); the Roboflow suffix `_jpg.rf.<hash>` is dropped from the file names. Boxes are clipped to the image. Each COCO image keeps `site`, `day`, `frame`, `recording` and `roboflow_split`. 328 frames without boxes are kept
- **Missing boxes**: in a hand count on 64 frames (6-8 per recording, boxes over ~3 cm only) the human boxes miss 29% of the animals (33-36% in 4 recordings, none in the 17 Apr ones), mostly gobies in busy frames and animals on the algae beside the mat. Only the original human boxes are used
- **Category mapping**:
  - fish: `Gobiidae sp` (1,443), `Pleuronectiformes sp` (14)
  - non-fish: `Portunoidea sp` (83, swimming and shore crabs), `Paguridae sp` (69, hermit crabs), `Hyas araneus` (55, spider crab)
  - discard: `Gastropoda sp` (265; pale snails of 2-3 cm crawling on the mat in the 17 Apr recordings: size doubtful, crops checked)
- **Split**: By sampling day (each day has one farm and one control recording; the Roboflow split is random across frames of the same recording). 4 May -> val (239 frames, 26%; 235 fish / 83 non-fish boxes); 6 Mar, 12 Apr and 17 Apr -> train (671 frames; 1,222 / 124). 6 Mar is almost only gobies, 12 Apr holds every hermit crab and 17 Apr only snails, so 4 May is the day with both fish and crabs

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
