# Image-edit Data Preparation


| Tasks            | Instruction Generation |
|------------------|------------------------|
| Add              | Need                   |
| Remove           | Need                   |
| Background       | Need                   |
| Text Edit        | Need                   |
| Segmentation     | -                      |
| Pose             | -                      |
| Detection        | -                      |
| Depth Maps       | -                      |
| Style Transfer   | Need                   |
| Lighting Control | Need                   |

## 1. Instruction Generation

### Environment

```
pip install torch==2.5.1 torchvision==0.20.1
pip install transformers==4.49.0
pip install safetensors qwen_vl_utils accelerate
pip install liger_kernel einops
```

### Examples

* Add
```py
from instruction_generation import instruction_generation

# Input
prompt = "A glowing blue sphere with floating metallic rings"
task = 'add'
pipe = instruction_generation(task=task)
for i in range(9):
    instruction = pipe.generate(prompt, task=task)
    print (instruction)

# Output
"add a colorful rainbow."
"make a robot stand next to the sphere"
"put a blue planet in the background"
"make the sphere have a yellow ring."
"add a dog and a cat."
"Set a firework in the back"
"add a flying saucer"
"Make it wear a helmet"
"make the sphere a little bit bigger"
```

* Lighting Control
  
```py
from instruction_generation import instruction_generation
# Input 
task='lighting'
pipe = instruction_generation(task=task)
image = "/mnt/pollux/wentian/image_edit/images/night.png"
instruction = pipe.generate(image, prompt=None, num_inst=5, task=task)
print (instruction)

image = "/mnt/pollux/wentian/image_edit/images/tree.png"
instruction = pipe.generate(image, prompt=None, num_inst=5, task=task)
print (instruction)

# Output:
['Increase the overall brightness of the image.', 
'Add more ambient light to the background.', 
'Brighten the faces of the individuals for better visibility.', 
'Enhance the lighting on the person sitting in the center.', 
'Soften the shadows around the group to create a more balanced look.'
]

['Brighten the entire scene slightly.', 
"Increase the lighting on the child's face.", 
'Enhance the glow around the sun.', 
"Soften the shadows on the mother's silhouette.", 
'Add a warm filter to the sunset colors.'
]
```

## 2. Image Generation

### Environment

```
conda create -n step1xedit python==3.10
conda activate step1xedit

pip install torch==2.5.1 torchvision==0.20.1
pip install transformers==4.49.0
pip install safetensors qwen_vl_utils accelerate
pip install liger_kernel einops

git clone https://github.com/stepfun-ai/Step1X-Edit.git

python scripts/get_flash_attn.py
https://github.com/Dao-AILab/flash-attention/releases
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

cp Step1X-Edit/sampling.py ./
cp -r Step1X-Edit/modules/ ./
```

### Examples

* Background

```py
from step1x_edit import ImageGenerator

prompt = 'A cute creature sits at the beach.'
image_path = '/mnt/pollux/wentian/image_edit/duck.jpeg'
instruction = "change the beach to a snowy mountain landscape"
image_edit = ImageGenerator(
    ae_path='/mnt/pollux/checkpoints/Step1X-Edit/vae.safetensors',
    dit_path='/mnt/pollux/checkpoints/Step1X-Edit/step1x-edit-i1258.safetensors',
    qwen2vl_model_path='/mnt/pollux/checkpoints/Qwen2.5-VL-7B-Instruct/',
    max_length=640,
    quantized=False,
    offload=False,
)
save_path = f'/mnt/pollux/wentian/image_edit/duck_edit_0.png'
image = image_edit.generate_image(
    instruction,
    negative_prompt="",
    ref_images=Image.open(image_path).convert("RGB"),
    num_samples=1,
    num_steps=28,
    cfg_guidance=2.0,
    seed=42,
    show_progress=True,
    size_level=1024,
)[0]
image.save(save_path, lossless=True)
```

* Style Transfer
  
```py
from step1x_edit import ImageGenerator

prompt = ['A photo of a zebra in the snow with a futuristic, cyberpunk aesthetic.', 
         'A photo of a zebra in the snow with a vintage, old-fashioned filter.', 
         'A photo of a zebra in the snow with a high-contrast, black-and-white effect.'
         ]
image_path = '/mnt/pollux/wentian/image_edit/images/zebra.png'

image_edit = ImageGenerator(
    ae_path='/mnt/pollux/checkpoints/Step1X-Edit/vae.safetensors',
    dit_path='/mnt/pollux/checkpoints/Step1X-Edit/step1x-edit-i1258.safetensors',
    qwen2vl_model_path='/mnt/pollux/checkpoints/Qwen2.5-VL-7B-Instruct/',
    max_length=640,
    quantized=False,
    offload=False,
)
for i in range(len(prompt)):
    save_path = f'/mnt/pollux/wentian/image_edit/images/zebra_edit_{str(i)}.png'
    image = image_edit.generate_image(
        prompt[i],
        negative_prompt="",
        ref_images=Image.open(image_path).convert("RGB"),
        num_samples=1,
        num_steps=28,
        cfg_guidance=6.0,
        seed=42,
        show_progress=True,
        size_level=1024,
    )[0]
    image.save(save_path, lossless=True)
```

## 3. Detection

Support Tasks: Object detection, Segmentation, and Pose Estimation.

### Environment
```
pip install ultralytics
```

### Examples

```py
from detection import Yolo_Detection
pipe = Yolo_Detection(task='detection')
pipe.pred(['/mnt/pollux/wentian/image_edit/images/desk.jpg'])

pipe = Yolo_Detection(task='segmentation')
pipe.pred(['/mnt/pollux/wentian/image_edit/images/desk.jpg'])

pipe = Yolo_Detection(task='pose')
pipe.pred(['/mnt/pollux/wentian/image_edit/images/pose.jpg'])
```
## 4. Depth Maps

### Environment
```
pip install transformers
```

### Examples
```py
from depth_detection import Depth_Pro

image_ = ['/mnt/pollux/wentian/image_edit/images/tiger.jpg']
model = Depth_Pro()
model.generate(image_,'/mnt/pollux/wentian/image_edit/')
```
