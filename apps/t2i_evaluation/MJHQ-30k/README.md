# MJHQ-30K

## Data

Download data 
```
wget https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json
wget https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/mjhq30k_imgs.zip
unzip mjhq30k_imgs.zip
```

Prepare image data

```
find <MJHQ30K_IMAGE_FOLDER> -name "*.jpg" -exec cp {} <MJHQ30K_IMAGE_FOLDER_SP> \;
```

## Install

* Create the environment and install the required packages:

```
conda create -n mjhq python=3.9 -y -c anaconda
conda activate mjhq
```

* Install dependencies

```
pip install torch torchvision torchaudio
pip install open-clip-torch
pip install glob
pip install tqdm
pip install matplotlib
pip install pytorch-fid
```

## Usage

### Calculate FID

Compute stats of MJHQ-30k images. (only need to execute once)
```
python -m pytorch_fid --save-stats <MJHQ30K_IMAGE_FOLDER_SP> <STATS_SAVE_FOLDER>/mjhq_30k_imgs.npz
```

Evaluation

```
python -m pytorch_fid <GENERATION_RESULT_FOLDER> <STATS_SAVE_FOLDER>/mjhq_30k_imgs.npz
```

For example,

```bash
python -m pytorch_fid /mnt/pollux/checkpoints/haozhe/qwen2_5_vl_flux_dynamic_max_ratio_1.0/MJHQ/0000040000/samples /mnt/pollux/mjhq_30k/mjhq_30k_imgs.npz
```

### Calculate CLIP score


```
python mjhq_clip.py --data_path '<GENERATION_RESULT_FOLDER>' ;
```

For example,

```bash
python mjhq_clip.py --data_path '/mnt/pollux/checkpoints/haozhe/qwen2_5_vl_flux_dynamic_max_ratio_1.0/MJHQ/0000040000/samples'
```



