### Install

``` sh
conda deactivate
conda remove -n videogen --all
conda create -n videogen python=3.11
conda activate videogen
```

### Run demo


``` sh
python generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir /mnt/pollux/mczhuge/Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```
