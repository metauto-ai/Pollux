# Install

This guide assumes Linux + NVIDIA GPUs.

## Prerequisites
- CUDA 12.8 (see `FA3.md`)
- Conda
- Python 3.12.9

## Environment
```bash
conda create -n pollux python=3.12.9
conda activate pollux
```

## PyTorch (CUDA 12.8)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Dependencies
```bash
pip install xformers ninja packaging
pip install -r requirements.txt
```

## FlashAttention (optional / recommended)
- For FlashAttention v3 build instructions, see `FA3.md`.
- Quick install:
```bash
MAX_JOBS=128 python -m pip -v install flash-attn --no-build-isolation
```

## COSMOS Tokenizer (optional)
```bash
cd apps/Cosmos-Tokenizer
pip3 install -e .
cd ../..
python apps/main/test_vae.py
```

## CLIP (optional)
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Extra deps (optional)
```bash
pip install timm torchmetrics
```

## MongoDB tools (optional)
If you need `mongoexport` / `mongoimport`, install the MongoDB database tools via your distro package manager.

Example (Ubuntu):
```bash
sudo apt-get update
sudo apt-get install mongodb-database-tools
```

Use environment variables (see `.env.sample`) and avoid hardcoding credentials:
```bash
mongoexport --uri="mongodb+srv://${MONGODB_USER}:${MONGODB_PASSWORD}@${MONGODB_URI}" --db=world_model --collection=COLLECTION --out=out.json --jsonArray
mongoimport --uri="mongodb+srv://${MONGODB_USER}:${MONGODB_PASSWORD}@${MONGODB_URI}" --db=world_model --collection=COLLECTION --file=in.json --jsonArray
```
