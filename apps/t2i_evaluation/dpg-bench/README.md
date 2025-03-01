# DPG-Bench


## Install

* Create the environment and install the required packages:

```
conda create -n dpgbench python=3.9 -y -c anaconda
conda activate dpgbench
```

* Install packages

```
python -m pip install pip==24.0
pip install -r requiements.txt
```

## Usage

### Grid image

DPG-Bench needs to generate 4 images and grid them to a 2x2 image per prompt for evaluation. 

```
python grid_image.py \
--result_csv_path '<GENERATION_RESULT_FOLDER>' \
--prompt_map_path './dpg_prompt_map.csv' \
--grid_save_path '<GRID_IMAGE_FOLDER>';
```

### Evaluation

```
accelerate launch --num_machines 1 --num_processes 4 --multi_gpu --mixed_precision "fp16" --main_process_port 29501 compute_dpg_bench.py \
--image-root-path '<GRID_IMAGE_FOLDER>' \
--csv 'dpg_bench.csv' \
--resolution <IMAGE RESOLUTION> \
--pic-num 4 \
--vqa-model mplug ;
```

The evaluation results are saved in `image-root-path` extended with `'.txt', '_detail.txt'`.
