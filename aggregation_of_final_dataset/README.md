# Aggregation of Final Dataset
This directory uses all the scripts in `data_preview` to load the different 
datasets and merge them into one single COCO dataset.

> Each script processes one datasets and outputs a train and val COCO dataset

We split each dataset into training and validation, ensuring each video/location is only contained in one of the two splits.

For this reason, we ultimately create two COCO datasets, one for training and one for validation.

## Usage
To use these files, firstly change the configs in `settings.py` to what fits your development environment, and then use the python modules.