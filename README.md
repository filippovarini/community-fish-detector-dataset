# Datasets with annotated fish in marine/freshwater imagery/video

## TOC

* <a href="#overview">Overview</a>
* <a href="#contributing-a-new-dataset">Contributing a new dataset</a>
* <a href="#publicly-available-datasets">Publicly available datasets</a>
* <a href="#skipped-datasets">Skipped datasets</a>
* <a href="#unprocessed-datasets">Unprocessed datasets</a>

## Overview

This is a list of datasets with annotated marine/freshwater imagery, suitable for training fish detectors/classifiers. 

It was suggested by [Dan Morris](https://dmorris.net) to emulate the [list of datasets with annotated wildlife in drone/aerial images](https://github.com/agentmorris/agentmorrispublic/blob/main/drone-datasets.md), with standardized metadata for each dataset, and consistent sample code for match annotations to images and rendering sample images.

This effort was supported by the following folks:
* <a href="https://www.linkedin.com/in/filippo-varini/">Filippo Varini</a>
* <a href="https://dmorris.net">Dan Morris</a>
* <a href="https://www.mbari.org/person/kevin-barnard/">Kevin Barnard</a>
* <a href="https://www.mbari.org/person/laura-chrobak/">Laura Chrobak</a>
* <a href="https://www.oceaneboulais.net/">Oceane Boulais</a>
* <a href="https://alexvmt.github.io/">Alexander Merdian-Tarko</a>
* <a href="https://www.linkedin.com/in/kameswari-devi-ayyagari-031820b7/">Devi Ayyagari</a>
* <a href="https://www.linkedin.com/in/sonny-burniston/">Sonny Burniston</a>
* <a href="https://www.linkedin.com/in/mona-dhiflaoui/">Mona Dhiflaoui</a>
* <a href="https://www.linkedin.com/in/jiashu-chen-w/">Joshua Chen</a>

Email <a href="mailto:fppvrn@gmail.com">Filippo</a> if anything seems off, or if you know of datasets we're missing.


## Contributing a new dataset

We welcome contributions! If you know of an underwater fish dataset that isn't listed here, you can help by processing it and submitting a pull request. Check the <a href="#unprocessed-datasets">Unprocessed datasets</a> section at the bottom of this README for datasets we already know about but haven't processed yet — picking one from that list is a great way to start.

### Acceptance criteria

A dataset must meet the following requirements to be included:

* **Underwater imagery** — images must be captured below the water surface (above-water and aerial fish images are rejected)
* **Contains fish** — the dataset must include annotations (bounding boxes or segmentation masks) on fish or fish-like organisms
* **Publicly available** — the data must be downloadable without requiring special access or paid subscriptions

### Processing rules

All datasets are normalized to a common format before merging. Your processing script must apply the following:

1. **Single category** — all fish-related annotations must be compressed into a single category: `{"id": 1, "name": "fish"}`. Regardless of how many species or sub-categories the source dataset has, we merge them all into one. Use `compress_annotations_to_single_category()` from `datasets/utils/`.
2. **1-indexed annotations** — COCO category and annotation IDs must be 1-indexed (not 0-indexed). Use `convert_coco_annotations_from_0_indexed_to_1_indexed()` if needed.
3. **Filter out non-fish categories** — if the source dataset contains non-fish categories (e.g. coral, crab, starfish), filter them out and keep only fish-related annotations. Define a `CATEGORIES_FILTER` list in your script to specify which source categories to keep.
4. **Prefix image filenames** — all image filenames must be prefixed with the dataset shortname (e.g. `noaa_puget_000001.jpg`) to avoid filename collisions when datasets are merged. Use `add_dataset_shortname_prefix_to_image_names()`.
5. **Train/val split** — split the dataset into training and validation sets. When possible, split by location, camera, video, or deployment rather than by random image selection. Use `split_coco_dataset_into_train_validation()`.
6. **COCO output format** — the final annotations must be in COCO format with bounding boxes.

### How to contribute

1. **Install dependencies**: `pip install -r requirements.txt`

2. **Create a script** at `datasets/<dataset_name>.py` that follows the 4-step pattern used by all other dataset scripts:
   - **Download** — download and extract the raw data to `data/raw/<dataset_name>/`
   - **Process** — convert annotations to COCO format, apply the processing rules above, and save to `data/processing/<dataset_name>/`
   - **Preview** — generate a sample annotated image and save it to `previews/`
   - **Split** — split into train/val and save to `data/final/<dataset_name>_train/` and `data/final/<dataset_name>_val/`

3. **Use shared utilities** from `datasets/utils/` — see existing scripts like <a href="./datasets/roboflow_fish.py">roboflow_fish.py</a> for a straightforward example.

4. **Define module-level constants**:
   - `DATASET_SHORTNAME` — a short, unique identifier (e.g. `"noaa_puget"`)
   - `CATEGORIES_FILTER` — list of source category names to keep (or `None` if all categories are fish)

5. **Add a dataset entry** to this README under <a href="#publicly-available-datasets">Publicly available datasets</a>, following the same metadata format as the existing entries.

6. **Submit a pull request** with your script, the preview image, and the README update.




### NOAA Puget Sound Nearshore Fish 2017-2018

Images with 67,990 bounding boxes on fish and crustaceans
  
Farrell DM, Ferriss B, Sanderson B, Veggerby K, Robinson L, Trivedi A, Pathak S, Muppalla S, Wang J, Morris D, Dodhia R. A labeled data set of underwater images of fish and crab species from five mesohabitats in Puget Sound WA USA. Scientific Data. 2023 Nov 13;10(1):799.

* Data downloadable via via https from LILA (<a href="http://lila.science/wp-content/uploads/2022/07/noaa-estuary-thumb-800.png">download link</a>)
* License: CDLA-permissive 1.0
* Metadata raw format: COCO
* Categories/species: fish and crustaceans
* Vehicle type: N/A
* Image information: 77,739 images
* Annotation information: 67,990 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/noaa_puget.py">noaa_puget.py</a>
  
<img src="./previews/noaa_puget_sample_image.png" width=700>

### MIT Sea Grant River Herring

Images of freshwater fish taken from underwater videos with 91,482 bounding boxes
  
[Project Website](https://www.woodwellclimate.org/project/fisheye/)

* Data downloadable via via https from LILA (<a href="https://lila.science/datasets/mit-sea-grant-river-herring/">download link</a>)
* License: CDLA-permissive 1.0
* Metadata raw format: COCO
* Categories/species: fish
* Vehicle type: Frames taken fromUnderwater Videos
* Image information: 262,050 images
* Annotation information: 91,482 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/mit_river_herring.py">mit_river_herring.py</a>
  
<img src="./previews/mit_river_herring_sample_image.png" width=700>

### Tasmanian Orange Roughy Stereo Image Machine Learning Dataset (TORSI)

Annotated stereo imagery of orange roughy from 2019 Tasmanian survey, with expert-labeled bounding boxes for machine learning detection in fisheries science.
  
Scoulding, Ben; Maguire, Kylie; Orenstein, Eric; Jackett, Chris; & CSIRO (2025): Tasmanian Orange Roughy Stereo Image Machine Learning Dataset. v1. CSIRO. Data Collection. https://doi.org/10.25919/a90r-4962

* Data downloadable via via https from the CSIRO Portal (<a href="https://data.csiro.au/collection/64913">download link</a>)
* License: CC BY-NC-SA 4.0
* Metadata raw format: COCO
* Categories/species: fish, eel, corals and other benthic organisms.
* Vehicle type: Net-attached Acoustic and Optical System (AOS)
* Image information: 1,051 images
* Annotation information: 14,414 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/torsi.py">torsi.py</a>
  
<img src="./previews/torsi_sample_image.png" width=700>
  

### CoralScapes

2,027 images captured by diver-borne GoPro cameras from a variety of global coral reefs.
  
* Home: <https://josauder.github.io/coralscapes/>  
* Data downloadable via HuggingFace from <https://huggingface.co/datasets/EPFL-ECEO/coralscapes>
* License: Apache-2.0
* Metadata raw format: Parquet
* Categories/species: fish, coral, human, rock, etc.
* Vehicle type: diver
* Image information: 2,027 images
* Annotation information: 174,000 segmentation annotations, of which 20,849 are fish
* Code to render sample annotated image: <a href="./datasets/coralscapes.py">coralscapes.py</a>
  
<img src="./previews/coralscapes-sample.jpg" width=700>

  
### Project Natick Underwater Video

~1k images of fish/squid w/bounding boxes
  
Simon, K. (2018). Project Natick - Microsoft's Self-sufficient Underwater Datacenters. IndraStra Global, 4(6), 1-4.
https://nbn-resolving.org/urn:nbn:de:0168-ssoar-57615-2

* Data downloadable via via https from GitHub (<a href="https://github.com/Microsoft/Project_Natick_Analysis/releases/tag/annotated_data">download link</a>)
* Metadata raw format: Pascal VOC
* Categories/species: fish, squid
* Vehicle type: fixed camera on structure
* Image information: 1118 RGB images (~5% of images have FN annotations)
* Annotation information: 998 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/project_natick.py">project_natick.py</a>
  
<img src="./previews/project_natick_sample_image.png" width=700>
  
  
### Roboflow Fish Dataset

~1k images of fish w/bounding boxes
  
Solawetz, J. (2023, February 21). Fish object detection dataset. Roboflow. https://public.roboflow.com/object-detection/fish 

* Data downloadable via via https from Roboflow (<a href="https://public.roboflow.com/object-detection/fish/1">download link</a>)
* License: CC0 1.0 DEED
* Metadata raw format: multiple available
* Categories/species: 26 fish types (e.g. shark, tuna)
* Vehicle type: underwater cameras
* Image information: 1350 RGB images (the taxonomy is often inaccurate)
* Annotation information: 3142 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/roboflow_fish.py">roboflow_fish.py</a>
  
<img src="./previews/roboflow_fish_sample_image.png" width=400>
  
  
### DeepFish

~40k images with a mix of classification, segmentation, and counting labels
  
Saleh A, Laradji IH, Konovalov DA, Bradley M, Vazquez D, Sheaves M. A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis. Scientific Reports. 2020 Sep 4;10(1):14671.

* Data downloadable via https from Queensland University (<a href="https://alzayats.github.io/DeepFish/">download link</a>) (7.1GB)
* License: Code is MIT, data is implied-MIT
* Metadata raw format: png (segmentation masks)
* Categories/species: 
* Vehicle type: underwater camera deployed over the side of a boat
* Image information: 311 images with segmentation masks
* Annotation information: 388 segmentation masks
* Code to render sample annotated image: <a href="./datasets/deepfish.py">deepfish.py</a>
  
<img src="./previews/deepfish_sample_image.jpg" width=700>  
  

### Deep Vision Fish Dataset

Bboxed images of pelagic fish and associated segmentations
  
Vaneeda Allken, Shale Rosen (2020) Deep Vision fish dataset https://doi.org/10.21335/NMDC-551736490

* Data downloadable via via https from the Norwegian Marine Data Centre (<a href="https://metadata.nmdc.no/metadata-api/landingpage/01d102345aef4639f063a13ea20cd3f3">download link</a>)
* License: CC BY 4.0
* Metadata raw format: csv
* Categories/species: economically important pelagic species
* Vehicle type: pictures from fish tanks
* Image information: 1875 RGB images
* Annotation information: 4834 bounding boxes, segmentation masks
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/deep_vision.py">deep_vision.py</a>
  
<img src="./previews/deep_vision_sample_image.png" width=700>
  

### The Brackish Dataset

~90 videos with bounding boxes on fish.  Largely redundant with BrackishMOT (see above).
  
Detection of Marine Animals in a New Underwater Dataset with Varying Visibility, Pedersen, Malte and Haurum, Joakim Bruslund and Gade, Rikke and Moeslund, Thomas B. and Madsen, Niels, June, 2019

* Data downloadable via via https from Kaggle (<a href="https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset">download link</a>)
* License: CC BY-SA 4.0
* Metadata raw format: AAU, COCO, YOLO
* Categories/species: fish, small fish, crab, shrimp, jellyfish, starfish
* Vehicle type: underwater cameras in brackish water
* Image information: 12,444 RGB images
* Annotation information: 35,565  bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/brackish.py">brackish.py</a>
  
<img src="./previews/brackish_dataset_sample_image.png" width=700>
  
### F4K Detection and Tracking

17 10-minute videos with tracking points
  
I. Kavasidis, S. Palazzo, R. Di Salvo, D. Giordano, C. Spampinato, An innovative web-based collaborative platform for video annotation, Multimedia Tools and Applications, vol. 70, pp. 413-432, 2013.

I. Kavasidis, S. Palazzo, R. Di Salvo, D. Giordano, C. Spampinato, A semi-automatic tool for detection and tracking ground truth generation in videos, Proceedings of the 1st International Workshop on Visual Interfaces for Ground Truth Collection in Computer Vision Applications, pp. 6:1-6:5, 2012.

* Data downloadable via https from GitHub (<a href="https://github.com/perceivelab/f4k-detection-and-tracking">download link</a>)
* Metadata raw format: XML, FLV
* Categories/species: N/A
* Vehicle type: N/A
* Image information: N/A
* Annotation information: N/A
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/f4k.py">f4k.py</a>

<img src="./previews/f4k_detection_tracking_sample.jpg" width=700>
  
  
### FishCLEF-2015 

14k boxes on fish in 20k images
  
Joly A., Goeau H., Glotin H., Spampinato C., Bonnet P., Vellinga W.-P., Planquè R., Rauber A., Palazzo S., Fisher R., and others}, LifeCLEF 2015: multimedia life species identification challenges, International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 462-483, Springer, 2015.

* Data downloadable via https from Zenodo (<a href="https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1">download link</a>). Note, the dataset was [originally hosted on SharePoint](https://github.com/perceivelab/FishCLEF-2015). We uploaded it to Zenodo to make it downloadable programmatically.
* Metadata raw format: XML
* Categories/species: marine ray-finned fish 
* Vehicle type: N/A
* Image information: 20,000 images
* Annotation information: 14,000 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/fishclef.py">fishclef.py</a>
  
<img src="./previews/fishclef_sample_image.png" width=700>
  
  
### VIAME FishTrack

Several thousand BRUV images with bounding boxes on fish and bait
  
* Data downloadable from Viame (<a href="https://viame.kitware.com/#/collection/65a140e8a4c218785d408b42">download link</a>)
* Metadata raw format: N/A
* Categories/species: General Fish, Fish Species, Bait and Algae
* Vehicle type: BRUV
* Image information: ~20,000
* Annotation information: bounding boxes
* Code to render sample annotated image: <a href="./datasets/viame_fishtrack.py">viame_fishtrack.py</a>

<img src="./previews/viame_fishtrack_sample_image.png" width=700>
  
  
### Object detection of tropical freshwater fish in Australia (Kakadu)

~44k images of fish w/ ~83kbounding boxes
  
Jansen, A., Walden, D., Walker, S., & Buccella, C. (2022). A deep learning dataset for underwater object detection of tropical freshwater fish species in northern Australia (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7250921

* Data downloadable via https from zenodo (<a href="https://zenodo.org/records/7250921#.ZEAmZezMJqs">download link</a>)
* License: CC BY 4.0 LEGAL CODE
* Metadata raw format: json
* Categories/species: Ambassis agrammus, Ambassis macleayi, Amniataba percoides, Craterocephalus stercusmuscarum, Denariusa bandata, Glossamia aprion, Glossogobius spp., Hephaestus fuliginosus, Lates calcarifer, Leiopotherapon unicolor, Liza ordensis, Megalops cyprinoides, Melanotaenia nigrans, Melanotaenia splendida inornata, Mogurnda mogurnda, Nemetalosa erebi, Neoarius spp., Neosilurus spp., Oxyeleotris spp., Scleropages jardinii, Strongylura kreffti, Syncomistes butleri, Toxotes chatareus
* Vehicle type: RUV
* Image information: 44,112 images (images were derived from Remote Underwater Video (RUV) deployments in deep channel and shallow lowland billabongs, Kakadu National Park, Northern Territory Australia)
* Annotation information: 82,904 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/kakadu.py">kakadu.py</a>

<img src="./previews/kakadu_sample_image.png" width=700>
  

### AAU Zebrafish Re-Identification Dataset

2200 images of zebrafish with individual IDs
  
Bruslund Haurum J, Karpova A, Pedersen M, Hein Bengtson S, Moeslund TB. Re-identification of zebrafish using metric learning. InProceedings of the IEEE/CVF winter conference on applications of computer vision workshops 2020 (pp. 1-11).

* Data downloadable via https from Kaggle (<a href="https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid">download link</a>)
* License: CC BY 4.0 DEED
* Metadata raw format: csv
* Categories/species: zebrafish
* Vehicle type: underwater camera in fish tank
* Image information: 2224 images
* Annotation information: AAU VAP bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/zebrafish.py">zebrafish.py</a>
  
<img src="./previews/Zebrafish_sample_image.png" width=700>
  
  
### Orange Chromide Pond Fish Detection

586 annotated underwater images of Orange Chromide (Etroplus maculatus) fish in South Indian pond environments with 10,607 bounding boxes

Vijayalakshmi M, Sasithradevi A (2024). Annotated underwater fish detection dataset from pond environments. Mendeley Data, V1. https://doi.org/10.17632/7w45jx35hd.1

* Data downloadable via https from Mendeley Data (<a href="https://data.mendeley.com/datasets/7w45jx35hd/1">download link</a>)
* License: CC BY 4.0
* Metadata raw format: YOLO TXT
* Categories/species: Orange Chromide (Etroplus maculatus)
* Vehicle type: Crosstour CT9000 underwater camera at <4m depth
* Image information: 586 images (640x640)
* Annotation information: 10,607 bounding boxes
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/orange_chromide.py">orange_chromide.py</a>

<img src="./previews/orange_chromide_sample_image.png" width=700>


### FathomNet Database

Katija, K., Orenstein, E., Schlining, B. _et al._ FathomNet: A global image database for enabling artificial intelligence in the ocean. Sci Rep 12, 15914 (2022). [https://doi.org/10.1038/s41598-022-19939-2](https://doi.org/10.1038/s41598-022-19939-2)

The FathomNet Database is an open-source image database that can be used to train, test, and validate state-of-the-art artificial intelligence algorithms to help us understand our ocean and its inhabitants. 

* Data downloadable via https from FathomNet (<a href="https://fathomnet.org/">website</a>)
* License: CC BY-NC-ND 4.0 with provision: _Notwithstanding any contrary provisions of such license, all Images may be used for training and development of machine learning algorithms for commercial, academic, and government purposes. For all other uses of the Images, users should contact the original copyright holder indicated in the Database for the applicable Images. The users of the Images accept full responsibility for their use._ (<a href="https://fathomnet.org/fathomnet/#/license">source</a>)
* Metadata raw format: N/A (exportable via [fathomnet-py](https://github.com/fathomnet/fathomnet-py) to VOC, COCO, YOLO)
* Categories/species: many (2400+ concepts in the database)
* Vehicle type: primarily ROV
* Image information: 100k+ images (growing over time)
* Annotation information: 300k+ annotations (growing over time)
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/fathomnet.py">fathomnet.py</a>

<img src="./previews/fathomnet_sample_image.png" width=700 />


### Marine Detect (FishInv + Megafauna)

Two merged Roboflow datasets with bounding boxes on fish, sharks, rays, turtles and other reef species

* Data downloadable via Roboflow (manual download required)
* Metadata raw format: COCO (after Roboflow export)
* Categories/species: fish, shark, ray, turtle, and various reef fish families
* Vehicle type: underwater cameras
* Code to render sample annotated image: <a href="./datasets/marine_detect.py">marine_detect.py</a>

<img src="./previews/marine_detect_fishinv_sample_image.png" width=700>

### Salmon Computer Vision

Boxes on 532,000 frames from 1,567 videos of salmon in two weirs
  
JOUR, Atlas, William, Ma, Sami, Chou, Yi, Connors, Katrina, Scurfield, Daniel, Nam, Brandon, Ma, Xiaoqiang, Cleveland, Mark, Doire, Janvier, Moore, Jonathan, Shea, Ryan, Liu, Jiangchuan, 2023/09/20, Wild salmon enumeration and monitoring using deep learning empowered detection and tracking, 10, 10.3389/fmars.2023.1200408, Frontiers in Marine Science

* Data downloadable via https from GitHub (<a href="https://github.com/Salmon-Computer-Vision/salmon-computer-vision">download link</a>)
* License: CC BY 4.0 
* Metadata raw format: YOLOv6
* Categories/species: pacific salmon
* Vehicle type: multi-object tracking (MOT) and object detection
* Image information: 1567 images
* Annotation information: bounding boxes
* Typical animal size in pixels: N/A
  
<img src="./previews/salmon_computer_vision_sample_image.jpg" width=700>


## Skipped datasets

Datasets that were evaluated but not included in the pipeline.

* **The Fishnet Dataset** (<a href="https://www.fishnet.ai/">link</a>) — ~163k bounding boxes on ~35k images of fish and people on fishing vessels. Rejected because above-water (may revisit in the future)
* **Visual Marine Animal Tracking (VMAT)** (<a href="https://link.springer.com/article/10.1007/s11263-023-01762-5#article-info">link</a>) — 32 video sequences with bounding boxes on marine organisms from AUVs. Decided to skip with Dan Morris and Sonny Burniston (meeting 28/04/2025).
* **BrackishMOT** — Same data as the Brackish dataset
* **Brackish Underwater Dataset** — Same data as the Brackish dataset
* **OzFish** — Annotations are of too poor quality
* **WildFish** — Images are already cropped, useless for training an object detector
* **Angling Freshwater Fish Netherlands (affine)** — Rejected as images are above water
* **Brook trout imagery for individual ID** — Rejected as images are above water
* **3D-ZeF20** — Rejected as images are above water and in lab environments
* **Croatian Fish** — Skipped
* **Application of a Deep Learning Image Classifier for Identification of Amazonian Fishes** — Skipped

## Unprocessed datasets

Datasets that we're aware exist, but that we haven't evaluated or processed yet.

* <a href="https://ouvert.canada.ca/data/dataset/2ae46860-f82a-4127-bb1f-b02e36ef6a70">Newfoundland Marine Refuge Fish Classification Dataset (N-MARINE)</a> (~24k images of marine fish in Canada, with ~24k boxes)
* <a href="https://zenodo.org/records/17432128">PomerFish: A dataset for fishes across Pomerania freshwater waterbodies in-situ environments</a> ([paper](https://www.nature.com/articles/s41597-025-06393-8)) (~20k segmentation masks on ~15k images)
