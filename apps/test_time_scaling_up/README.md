## Test-time Scaling up 

We use multiple SOTA test-time scaling up methods to score images from candidates to boost model performance.

## Environment Requirement

```
pip install glob
pip install tqdm
pip install pillow
```
#### QWEN-2.5-VL-7B Model Installation

```
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

#### UnifiedReward-7b-v1.5 Model Installation

```
cd unified_reward
pip install -e ".[train]"
```
#### Nvila_Lite-2B-Verifier Model Installation

```
pip install transformers==4.46 accelerate opencv-python torchvision einops pillow
pip install git+https://github.com/bfshi/scaling_on_scales.git
```

## Select images

The selected images and corresponding prompt are saved as CSV file in `save_path`.
```
CUDA_VISIBLE_DEVICES=0 python image_pick.py \
--result_csv_file '<GENERATION_RESULT_FOLDER_or_CSV_DIR>' \
--top_k <TOP_NUMBER_to_Select> \
--model <SCALING_METHOD_NAME> \ # 'qwen', 'nvila', 'unified_reward'
--save_path '<CSV_FILE_OUTPUT_FOLDER>' ;
```

