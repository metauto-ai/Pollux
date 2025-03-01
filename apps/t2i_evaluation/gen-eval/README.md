# GenEval-Bench


## Install

* Create the environment and install the required packages:

```
conda create -n geneval python=3.9 -y -c anaconda
conda activate geneval
```

* Install dependencies

```
pip install torch torchvision torchaudio
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
python -m pip install lightning
pip install diffusers["torch"] transformers
pip install tomli
pip install platformdirs
pip install --upgrade setuptools (You need to have the latest version)

mim install mmengine mmcv-full==1.7.2 (Should be done after pip installations)

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

* Download model

```
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "./mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
```

## Usage

### Evaluation

The evaluation results are saved as `'<OUTPUT_FOLDER>/results.jsonl'`.

```
CUDA_VISIBLE_DEVICES=1 python evaluate_images.py '<GENERATION_RESULT_FOLDER>' \
--result-csv-path '<GENERATION_RESULT_FOLDER>' \
--eval-metadata-dir 'evaluation_metadata.jsonl' \
--outfile '<OUTPUT_FOLDER>/results.jsonl' ;
```

### Summary

The summary score is printed by 

```
python summary_scores.py '<OUTPUT_FOLDER>/results.jsonl'
```



