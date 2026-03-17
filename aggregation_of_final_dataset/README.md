# How to build the final dataset
1. Create a virtual environment with `requirements.txt`
2. Change `raw_dir`, `processed_dir`, `intermediate_dir` in `aggregation_of_final_dataset/settings.py`
3. Run `export PYTHONPATH="$(pwd):PYTHONPATH"`
4. Aggregate each dataset by following the guides below

### Project Natick
1. Simply run `python aggregation_of_final_dataset/project_natick.py`

### Roboflow Fish
1. Simply run `python aggregation_of_final_dataset/roboflow_fish.py`

### Zebrafish
1. Simply run `python aggregation_of_final_dataset/zebrafish.py`

### NOAA Puget
1. Simply run `python aggregation_of_final_dataset/noaa_puget.py`

### Brackish Dataset
1. Simply run `python aggregation_of_final_dataset/brackish.py`

### Deepfish
1. Simply run `python aggregation_of_final_dataset/deepfish.py`

### Deep Vision
1. Simply run `python aggregation_of_final_dataset/deep_vision.py`

### Viame Fishtrack
1. Simply run `python aggregation_of_final_dataset/viame_fishtrack.py`

### Fishclef
1. Simply run `python aggregation_of_final_dataset/fishclef.py`
⚠️ One frame is not extracted properly from videos `01465f8f61db58564cd37ce2dfc519c5#201106090830_0_frame_1000.jpg` and thus causes a warning (no error). If you run the file you'll build successfully the dataset, and we'll exclude just 1 (out of ~15k) images.

### Fathomnet
1. Simply run `python aggregation_of_final_dataset/fathomnet.py`
⚠️ The `download_data` function takes ages (~12h) but it works

### MIT River Herring
1. Simply run `python aggregation_of_final_dataset/mit_river_herring.py`


## Aggregated dataset
- The `image_id` is the filename (unique)
- The `catecory_id` is monotonically increasing from 0