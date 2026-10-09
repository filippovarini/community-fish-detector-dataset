# Datasets with annotated fish in marine/freshwater imagery/video

## TOC

- [Overview](#overview)
- [Contributing a new dataset](#contributing-a-new-dataset)
- [Fish datasets](#fish-datasets)
  - [Processed datasets](#processed-datasets)
  - [Datasets that were added after the most recent CFD training](#datasets-that-were-added-after-the-most-recent-cfd-training)
  - [Skipped datasets](#skipped-datasets)
  - [Unprocessed datasets](#unprocessed-datasets)

## Overview

This repository contains a list of datasets with annotated marine/freshwater imagery and the scripts we used to process, clean and aggregate them to create the [Community Fish Detection (CFD) Dataset](https://lila.science/datasets/community-fish-detection-dataset/), which we used to train the [Community Fish Detector](https://github.com/filippovarini/community-fish-detector). 

This effort was supported by the following folks: [Filippo Varini](https://www.linkedin.com/in/filippo-varini/), [Dan Morris](https://dmorris.net), [Kevin Barnard](https://www.mbari.org/person/kevin-barnard/), [Laura Chrobak](https://www.mbari.org/person/laura-chrobak/), [Oceane Boulais](https://www.oceaneboulais.net/), [Alexander Merdian-Tarko](https://alexvmt.github.io/), [Devi Ayyagari](https://www.linkedin.com/in/kameswari-devi-ayyagari-031820b7/), [Sonny Burniston](https://www.linkedin.com/in/sonny-burniston/), [Mona Dhiflaoui](https://www.linkedin.com/in/mona-dhiflaoui/), [Joshua Chen](https://www.linkedin.com/in/jiashu-chen-w/)

Email [Filippo](mailto:fppvrn@gmail.com) if anything seems off, or if you know of datasets we're missing.

## Contributing a new dataset

We welcome contributions! If you know of an underwater fish dataset that isn't listed here, you can help by processing it and submitting a pull request. Check the [Unprocessed datasets](#unprocessed-datasets) section at the bottom of this README for datasets we already know about but haven't processed yet — picking one from that list is a great way to start.

### Acceptance criteria

A dataset must meet the following requirements to be included:

- **Underwater imagery** — images must be captured below the water surface (above-water and aerial fish images are rejected)
- **Contains fish** — the dataset must include annotations (bounding boxes or segmentation masks) on fish (see [Class definitions](#class-definitions))
- **Publicly available** — the data must be downloadable without requiring special access or paid subscriptions



### Class definitions

Every annotation in the final dataset belongs to one of two categories:

| id | name | Definition | Examples |
|----|------|------------|----------|
| 1 | `fish` | Any cartilaginous, ray-finned, or bony fish | sharks, rays, skates, tunas, groupers, eels, salmon, seahorses |
| 2 | `non-fish` | Any marine animal bigger than ~3 cm that is **not** a fish | turtles, dolphins, whales, seals, manatees, crabs, lobsters, shrimp, octopus, squid, jellyfish, starfish, sea urchins |

Anything that is neither (corals, algae, plants, rocks, debris, divers/humans, equipment, bait, animals smaller than ~3 cm) is discarded.

Note that some animals with "fish" in their name are **not** fish (jellyfish, starfish, cuttlefish, crayfish → `non-fish`), and some fish-shaped animals are **not** fish (dolphins, whales, manatees → `non-fish`).

**When contributing a dataset, you must list every class in the source annotations and assign each one to `fish`, `non-fish`, or discard.** Include this mapping in your script (see `CATEGORIES_FILTER` below) and in your pull request description, so reviewers can check it. Generic or ambiguous classes (e.g. "animal", "unknown", "other") should be inspected visually: assign them if they are consistently one or the other, otherwise discard them and mention it in the PR.



### Processing rules

All datasets are normalized to a common format before merging. Your processing script must apply the following:

1. **Map every source class** — assign each source category to `fish` (id 1), `non-fish` (id 2), or discard, following the [Class definitions](#class-definitions) above. Regardless of how many species or sub-categories the source dataset has, they all collapse into these two categories. Define the mapping at module level via `CATEGORIES_FILTER` in your script.
2. **1-indexed annotations** — COCO category and annotation IDs must be 1-indexed (not 0-indexed). Use `convert_coco_annotations_from_0_indexed_to_1_indexed()` if needed.
3. **Prefix image filenames** — all image filenames must be prefixed with the dataset shortname (e.g. `noaa_puget_000001.jpg`) to avoid filename collisions when datasets are merged. Use `add_dataset_shortname_prefix_to_image_names()`.
4. **Train/val split** — split the dataset into training and validation sets. When possible, split by location, camera, video, or deployment rather than by random image selection. Use `split_coco_dataset_into_train_validation()`.
5. **COCO output format** — the final annotations must be in COCO format with bounding boxes.



### How to contribute



#### Option A: Just suggest a dataset (no coding required)

If you've come across a dataset that matches the acceptance criteria above but don't have the time or experience to write a processing script, you can still help:

- Add it to the [Unprocessed datasets](#unprocessed-datasets) list at the bottom of this README via a pull request, or
- Email it to [Filippo](mailto:fppvrn@gmail.com) and we'll add it to the list



#### Option B: Write a processing script

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Create a script** at `datasets/<dataset_name>.py` that follows the 4-step pattern used by all other dataset scripts:
  - **Download** — download and extract the raw data to `data/raw/<dataset_name>/`
  - **Process** — convert annotations to COCO format, apply the processing rules above, and save to `data/processing/<dataset_name>/`
  - **Preview** — generate a sample annotated image and save it to `previews/`
  - **Split** — split into train/val and save to `data/final/<dataset_name>_train/` and `data/final/<dataset_name>_val/`
3. **Use shared utilities** from `datasets/utils/` — see existing scripts like [roboflow_fish.py](./datasets/roboflow_fish.py) for a straightforward example.
4. **Define module-level constants**:
  - `DATASET_SHORTNAME` — a short, unique identifier (e.g. `"noaa_puget"`)
  - `CATEGORIES_FILTER` — the mapping of every source category name to `fish` or `non-fish` (categories not listed are discarded)
5. **Add a dataset entry** to this README under [Processed datasets](#processed-datasets), following the same metadata format as the existing entries.
6. **Submit a pull request** with your script, the preview image, the README update, and the full list of source classes with the `fish` / `non-fish` / discard decision for each.



#### Option C: Use Claude Code to generate the processing script

This repo includes a custom [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill that automates the entire dataset processing workflow — from downloading and analyzing the data, to generating the script, running it, and updating the documentation. To use it:

1. Install Claude Code and open this repo
2. Paste the dataset URL (e.g. a Zenodo link, a paper, a GitHub repo) and type `/add-dataset <url>`
3. Claude Code will walk you through the full pipeline: research the dataset, test the download, write the processing script, run it, and update the docs

This is the fastest way to add a new dataset if you're already familiar with Claude Code.

## Fish datasets



### Datasets that were included in the Community Fish Detection Dataset



#### NOAA Puget Sound Nearshore Fish 2017-2018

Images with 67,990 bounding boxes on fish and crustaceans

Farrell DM, Ferriss B, Sanderson B, Veggerby K, Robinson L, Trivedi A, Pathak S, Muppalla S, Wang J, Morris D, Dodhia R. A labeled data set of underwater images of fish and crab species from five mesohabitats in Puget Sound WA USA. Scientific Data. 2023 Nov 13;10(1):799.

- Data downloadable via via https from LILA ([download link](http://lila.science/wp-content/uploads/2022/07/noaa-estuary-thumb-800.png))
- License: CDLA-permissive 1.0
- Metadata raw format: COCO
- Categories/species: fish and crustaceans
- Vehicle type: N/A
- Image information: 77,739 images
- Annotation information: 67,990 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [noaa_puget.py](./datasets/noaa_puget.py)



#### MIT Sea Grant River Herring

Images of freshwater fish taken from underwater videos with 91,482 bounding boxes

[Project Website](https://www.woodwellclimate.org/project/fisheye/)

- Data downloadable via via https from LILA ([download link](https://lila.science/datasets/mit-sea-grant-river-herring/))
- License: CDLA-permissive 1.0
- Metadata raw format: COCO
- Categories/species: fish
- Vehicle type: Frames taken fromUnderwater Videos
- Image information: 262,050 images
- Annotation information: 91,482 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [mit_river_herring.py](./datasets/mit_river_herring.py)



#### Tasmanian Orange Roughy Stereo Image Machine Learning Dataset (TORSI)

Annotated stereo imagery of orange roughy from 2019 Tasmanian survey, with expert-labeled bounding boxes for machine learning detection in fisheries science.

Scoulding B, Maguire K, Orenstein E, Jackett C, CSIRO.  [Tasmanian Orange Roughy Stereo Image Machine Learning Dataset](https://doi.org/10.25919/a90r-4962). v1. CSIRO. Data Collection. 2025.

- Data downloadable via via https from the CSIRO Portal ([download link](https://data.csiro.au/collection/64913))
- License: CC BY-NC-SA 4.0
- Metadata raw format: COCO
- Categories/species: fish, eel, corals and other benthic organisms.
- Vehicle type: Net-attached Acoustic and Optical System (AOS)
- Image information: 1,051 images
- Annotation information: 14,414 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [torsi.py](./datasets/torsi.py)



#### CoralScapes

2,027 images captured by diver-borne GoPro cameras from a variety of global coral reefs.

- Home: [https://josauder.github.io/coralscapes/](https://josauder.github.io/coralscapes/)  
- Data downloadable via HuggingFace from [https://huggingface.co/datasets/EPFL-ECEO/coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes)
- License: Apache-2.0
- Metadata raw format: Parquet
- Categories/species: fish, coral, human, rock, etc.
- Vehicle type: diver
- Image information: 2,027 images
- Annotation information: 174,000 segmentation annotations, of which 20,849 are fish
- Code to render sample annotated image: [coralscapes.py](./datasets/coralscapes.py)



#### Project Natick Underwater Video

~1k images of fish/squid w/bounding boxes

Simon K. [ProjectNatick - Microsoft's Self-sufficient Underwater Datacenters](https://nbn-resolving.org/urn:nbn:de:0168-ssoar-57615-2). IndraStra Global, 4(6), 1-4. 2018.

- Data downloadable via via https from GitHub ([download link](https://github.com/Microsoft/Project_Natick_Analysis/releases/tag/annotated_data))
- Metadata raw format: Pascal VOC
- Categories/species: fish, squid
- Vehicle type: fixed camera on structure
- Image information: 1118 RGB images (~5% of images have FN annotations)
- Annotation information: 998 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [project_natick.py](./datasets/project_natick.py)



#### Roboflow Fish Dataset

~1k images of fish w/bounding boxes

Solawetz J Fish object detection dataset. Roboflow. 2023. 

- Data downloadable via via https from Roboflow ([download link](https://public.roboflow.com/object-detection/fish/1))
- License: CC0 1.0 DEED
- Metadata raw format: multiple available
- Categories/species: 26 fish types (e.g. shark, tuna)
- Vehicle type: underwater cameras
- Image information: 1350 RGB images (the taxonomy is often inaccurate)
- Annotation information: 3142 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [roboflow_fish.py](./datasets/roboflow_fish.py)



#### DeepFish

~40k images with a mix of classification, segmentation, and counting labels

Saleh A, Laradji IH, Konovalov DA, Bradley M, Vazquez D, Sheaves M. A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis. Scientific Reports. 2020 Sep 4;10(1):14671.

- Data downloadable via https from Queensland University ([download link](https://alzayats.github.io/DeepFish/)) (7.1GB)
- License: Code is MIT, data is implied-MIT
- Metadata raw format: png (segmentation masks)
- Categories/species: 
- Vehicle type: underwater camera deployed over the side of a boat
- Image information: 311 images with segmentation masks
- Annotation information: 388 segmentation masks
- Code to render sample annotated image: [deepfish.py](./datasets/deepfish.py)



#### Deep Vision Fish Dataset

Bboxed images of pelagic fish and associated segmentations

Allken V, Rosen S. [Deep Vision fish dataset](https://doi.org/10.21335/NMDC-551736490). 2020.

- Data downloadable via via https from the Norwegian Marine Data Centre ([download link](https://metadata.nmdc.no/metadata-api/landingpage/01d102345aef4639f063a13ea20cd3f3))
- License: CC BY 4.0
- Metadata raw format: csv
- Categories/species: economically important pelagic species
- Vehicle type: pictures from fish tanks
- Image information: 1875 RGB images
- Annotation information: 4834 bounding boxes, segmentation masks
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [deep_vision.py](./datasets/deep_vision.py)



#### The Brackish Dataset

~90 videos with bounding boxes on fish.  Largely redundant with BrackishMOT (see above).

Pedersen M, Haurum JB, Gade R, Moeslund TB, Madsen N.  Detection of Marine Animals in a New Underwater Dataset with Varying Visibility. 2019.

- Data downloadable via via https from Kaggle ([download link](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset))
- License: CC BY-SA 4.0
- Metadata raw format: AAU, COCO, YOLO
- Categories/species: fish, small fish, crab, shrimp, jellyfish, starfish
- Vehicle type: underwater cameras in brackish water
- Image information: 12,444 RGB images
- Annotation information: 35,565  bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [brackish.py](./datasets/brackish.py)



#### F4K Detection and Tracking

17 10-minute videos with tracking points

Kavasidis I, Palazzo S, Di Salvo R, Giordano D, Spampinato C. An innovative web-based collaborative platform for video annotation, Multimedia Tools and Applications. 2013.

Kavasidis I, Palazzo S, Di Salvo R, Giordano D, Spampinato C. A semi-automatic tool for detection and tracking ground truth generation in videos. Proceedings of the 1st International Workshop on Visual Interfaces for Ground Truth Collection in Computer Vision Applications. 2012.

- Data downloadable via https from GitHub ([download link](https://github.com/perceivelab/f4k-detection-and-tracking))
- Metadata raw format: XML, FLV
- Categories/species: N/A
- Vehicle type: N/A
- Image information: N/A
- Annotation information: N/A
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [f4k.py](./datasets/f4k.py)



#### FishCLEF-2015

14k boxes on fish in 20k images

Joly A, Goeau H, Glotin H, Spampinato C, Bonnet P, Vellinga W-P, Planquè R, Rauber A, Palazzo S, Fisher R.  LifeCLEF 2015: multimedia life species identification challenges, International Conference of the Cross-Language Evaluation Forum for European Languages. 2015.

- Data downloadable via https from Zenodo ([download link](https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1)). Note, the dataset was [originally hosted on SharePoint](https://github.com/perceivelab/FishCLEF-2015). We uploaded it to Zenodo to make it downloadable programmatically.
- Metadata raw format: XML
- Categories/species: marine ray-finned fish 
- Vehicle type: N/A
- Image information: 20,000 images
- Annotation information: 14,000 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [fishclef.py](./datasets/fishclef.py)



#### VIAME FishTrack

Several thousand BRUV images with bounding boxes on fish and bait

- Data downloadable from Viame ([download link](https://viame.kitware.com/#/collection/65a140e8a4c218785d408b42))
- Metadata raw format: N/A
- Categories/species: General Fish, Fish Species, Bait and Algae
- Vehicle type: BRUV
- Image information: ~20,000
- Annotation information: bounding boxes
- Code to render sample annotated image: [viame_fishtrack.py](./datasets/viame_fishtrack.py)



#### Object detection of tropical freshwater fish in Australia (Kakadu)

~44k images of fish w/ ~83kbounding boxes

Jansen A, Walden D, Walker S, Buccella C.  [A deep learning dataset for underwater object detection of tropical freshwater fish species in northern Australia](https://doi.org/10.5281/zenodo.7250921) (dataset).  2022.

- Data downloadable via https from zenodo ([download link](https://zenodo.org/records/7250921#.ZEAmZezMJqs))
- License: CC BY 4.0 LEGAL CODE
- Metadata raw format: json
- Categories/species: Ambassis agrammus, Ambassis macleayi, Amniataba percoides, Craterocephalus stercusmuscarum, Denariusa bandata, Glossamia aprion, Glossogobius spp., Hephaestus fuliginosus, Lates calcarifer, Leiopotherapon unicolor, Liza ordensis, Megalops cyprinoides, Melanotaenia nigrans, Melanotaenia splendida inornata, Mogurnda mogurnda, Nemetalosa erebi, Neoarius spp., Neosilurus spp., Oxyeleotris spp., Scleropages jardinii, Strongylura kreffti, Syncomistes butleri, Toxotes chatareus
- Vehicle type: RUV
- Image information: 44,112 images (images were derived from Remote Underwater Video (RUV) deployments in deep channel and shallow lowland billabongs, Kakadu National Park, Northern Territory Australia)
- Annotation information: 82,904 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [kakadu.py](./datasets/kakadu.py)



#### AAU Zebrafish Re-Identification Dataset

2200 images of zebrafish with individual IDs

Bruslund HJ, Karpova A, Pedersen M, Hein BS, Moeslund TB. Re-identification of zebrafish using metric learning. InProceedings of the IEEE/CVF winter conference on applications of computer vision workshops. 2020.

- Data downloadable via https from Kaggle ([download link](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid))
- License: CC BY 4.0 DEED
- Metadata raw format: csv
- Categories/species: zebrafish
- Vehicle type: underwater camera in fish tank
- Image information: 2224 images
- Annotation information: AAU VAP bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [zebrafish.py](./datasets/zebrafish.py)



#### Orange Chromide Pond Fish Detection

586 annotated underwater images of Orange Chromide (Etroplus maculatus) fish in South Indian pond environments with 10,607 bounding boxes

Vijayalakshmi M, Sasithradevi A.  [Annotated underwater fish detection dataset from pond environments](https://doi.org/10.17632/7w45jx35hd.1). 2024

- Data downloadable via https from Mendeley Data ([download link](https://data.mendeley.com/datasets/7w45jx35hd/1))
- License: CC BY 4.0
- Metadata raw format: YOLO TXT
- Categories/species: Orange Chromide (Etroplus maculatus)
- Vehicle type: Crosstour CT9000 underwater camera at <4m depth
- Image information: 586 images (640x640)
- Annotation information: 10,607 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [orange_chromide.py](./datasets/orange_chromide.py)



#### FathomNet Database

Katija K, Orenstein E, Schlining B, et al. [FathomNet: A global image database for enabling artificial intelligence in the ocean](https://doi.org/10.1038/s41598-022-19939-2). Sci Rep 12, 15914 (2022).

The FathomNet Database is an open-source image database that can be used to train, test, and validate state-of-the-art artificial intelligence algorithms to help us understand our ocean and its inhabitants. 

- Data downloadable via https from FathomNet ([website](https://fathomnet.org/))
- License: CC BY-NC-ND 4.0 with provision: *Notwithstanding any contrary provisions of such license, all Images may be used for training and development of machine learning algorithms for commercial, academic, and government purposes. For all other uses of the Images, users should contact the original copyright holder indicated in the Database for the applicable Images. The users of the Images accept full responsibility for their use.* ([source](https://fathomnet.org/fathomnet/#/license))
- Metadata raw format: N/A (exportable via [fathomnet-py](https://github.com/fathomnet/fathomnet-py) to VOC, COCO, YOLO)
- Categories/species: many (2400+ concepts in the database)
- Vehicle type: primarily ROV
- Image information: 100k+ images (growing over time)
- Annotation information: 300k+ annotations (growing over time)
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [fathomnet.py](./datasets/fathomnet.py)



#### Marine Detect (FishInv and Megafauna)

Two Roboflow datasets with bounding boxes on fish, sharks, rays, turtles and other reef species

- Data downloadable via Roboflow (manual download required)
- Metadata raw format: COCO (after Roboflow export)
- Categories/species: fish, shark, ray, turtle, and various reef fish families
- Vehicle type: underwater cameras
- Code to render sample annotated image: [marine_detect.py](./datasets/marine_detect.py)



#### Salmon Computer Vision

Boxes on 532,000 frames from 1,567 videos of salmon in two weirs

Atlas WI, Ma S, Chou YC, Connors K, Scurfield D, Nam B, Ma X, Cleveland M, Doire J, Moore JW, Shea R. Wild salmon enumeration and monitoring using deep learning empowered detection and tracking. Frontiers in Marine Science. 2023 Sep 20.

- Data downloadable via https from GitHub ([download link](https://github.com/Salmon-Computer-Vision/salmon-computer-vision))
- License: CC BY 4.0 
- Metadata raw format: YOLOv6
- Categories/species: pacific salmon
- Vehicle type: multi-object tracking (MOT) and object detection
- Image information: 1567 images
- Annotation information: bounding boxes
- Typical animal size in pixels: N/A



### Datasets that were added after the most recent CFD training

#### PomerFish

Underwater video surveillance of freshwater fish species collected 2015–2024 in Pomeranian rivers using GoPro Hero 5 cameras.

Shi, X., Tanwari, K. A., Krepski, T., Shi, Z., & Czerniawski, R. (2025). PomerFish: A dataset for fishes across Pomerania freshwater waterbodies in-situ environments. Zenodo. [https://doi.org/10.5281/zenodo.17432128](https://doi.org/10.5281/zenodo.17432128)

- Data downloadable via HTTPS from Zenodo ([download link](https://zenodo.org/records/17432128/files/PomerFish.rar?download=1))
- License: CC-BY-4.0
- Metadata raw format: COCO JSON
- Categories/species: Perca fluviatilis, Thymallus thymallus, Salmo trutta (male/female), Salmoninae (Juvenile), Salmo trutta morpha fario (Adult/juvenile), Rutilus rutilus, Oncorhynchus mykiss, Leuciscus idus
- Vehicle type: GoPro Hero 5 (in-situ freshwater)
- Image information: 14,989 images
- Annotation information: 21,267 bounding boxes
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [pomerfish.py](./datasets/pomerfish.py)



#### OBSEA

Boxes on fish and other objects in images from the OBSEA seafloor observatory in the Mediterranean

- Data downloadable via Zenodo ([download link](https://doi.org/10.5281/zenodo.14888328))
- License: CC BY 4.0
- Metadata raw format: YOLO
- Categories/species: 74 source classes
- Vehicle type: fixed underwater observatory camera
- Image information: 4,120 images, including 470 zero-fish/background images
- Annotation information: 34,728 fish/fish-like bounding boxes after filtering
- Typical animal size in pixels: N/A
- Code to render sample annotated image: [obsea.py](./datasets/obsea.py)



### Datasets that were not included in the Community Fish Detection Dataset

- [Fishnet.AI](https://www.fishnet.ai/): ~163k bounding boxes on ~35k images of fish and people on fishing vessels.  Excluded because images are above-water.
- [Visual Marine Animal Tracking (VMAT)](https://link.springer.com/article/10.1007/s11263-023-01762-5#article-info): 32 video sequences with bounding boxes on marine organisms from AUVs.
- [OzFish](https://github.com/open-AIMS/ozfish): Excluded due to annotation quality issues.
- [WildFish](https://github.com/PeiqinZhuang/WildFish): Excluded because images are cropped.
- [Angling Freshwater Fish Netherlands (AFFiNe)](https://www.kaggle.com/datasets/jorritvenema/affine): 7k images of 30 species, excluded because images are above water.
- [Brook trout imagery for individual ID](https://www.usgs.gov/data/brook-trout-imagery-data-individual-recognition-deep-learning): Excluded because images are above water.
- [3D-ZeF](https://huggingface.co/datasets/vapaau/3D-ZeF): Excluded because images are above water and in lab environments.
- [Croatian Fish](https://www.kaggle.com/datasets/ashfaqsyed/croatian-fish-dataset): 794 images of 12 species, excluded because images are cropped.
- [Amazonian Fish ML Classifier](https://sidatasciencelab.github.io/Amazonian_Fish_ML_Classifier/): Excluded because images are in staging environments.
- [BrackishMOT](https://www.kaggle.com/datasets/maltepedersen/brackishmot): Same data as the Brackish Dataset.
- Brackish Underwater Dataset: Same data as the Brackish Dataset.



### Unprocessed datasets

Datasets that we're aware exist, but that we haven't evaluated or processed yet.

- [WIO-ReefFish](https://zenodo.org/records/21360120) (~6.8k boxes on 1k images with 24k labeled fish taxa from the Indian Ocean)
- [Newfoundland Marine Refuge Fish Classification Dataset (N-MARINE)](https://ouvert.canada.ca/data/dataset/2ae46860-f82a-4127-bb1f-b02e36ef6a70) (~24k images of marine fish in Canada, with ~24k boxes)
- [J-EDI](https://www.godac.jamstec.go.jp/jedi/e/dataset/jedi_organism_detection_dataset.html) (~8k images with 19 deep-sea animal categories)
- [OBSEA](https://zenodo.org/records/14888328) (~35k boxes on ~4k images from a cabled observatory in the Mediterranean)
- [FjordFish](https://zenodo.org/records/17950781) (~6k boxes on ~3k images from a BRUV in the North Atlantic)
- [SFISHTRACK](https://github.com/JosepSanchezCano/SFISHTRACK) (~24k frames with segmentation masks)
- [CoralscapesV2](https://josauder.github.io/coralscapesv2/) (~2.4k images with ~66k fish instance masks; Coralscapes v1 is already included in CFD)

