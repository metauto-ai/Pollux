import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer
from vargpt_qwen_v1_1.modeling_vargpt_qwen2_vl import VARGPTQwen2VLForConditionalGeneration
from vargpt_qwen_v1_1.prepare_vargpt_v1_1 import prepare_vargpt_qwen2vl_v1_1 
from vargpt_qwen_v1_1.processing_vargpt_qwen2_vl import VARGPTQwen2VLProcessor
from patching_utils.patching import patching
import argparse
from instruction_generation import Llama_Instruction



def parse_args():
    parser = argparse.ArgumentParser(description='Image Edit Instruction Generation', add_help=False)
    parser.add_argument('--task', type=str, default='add', help='which image edit task')
    parser.add_argument('--result_csv_file', type=str,help='path to <prompt path> csv')
    parser.add_argument('--save_path', metavar='DIR',help='path to save <prompt path> csv')
    parser.add_argument('--num_inst', type=int, default=1, help='num instruction per prompt')
    args = parser.parse_args()
    return args

def read_result_csv(csv_path):
    """
    Combine images with a same prompt as dict format
    Args:
        csv_path: the dir or file path of csv file containing data pair <prompt, image_path>
    Return: 
        dict {prompt: [image1, image2, image3],}
    """
    image_dict = {}
    if os.path.isfile(csv_path):
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for rows in reader:
                if (rows[0]) in image_dict:
                    image_dict[str(rows[0])].append(str(rows[1]))  
                else:
                    image_dict[str(rows[0])] = [str(rows[1])]

    elif os.path.isdir(csv_path):
        csv_dir_list = glob.glob(csv_path+'*.csv')
        for csv_dir in csv_dir_list:
            with open(csv_dir, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for rows in reader:
                    if (rows[0]) in image_dict:
                        image_dict[str(rows[0])].append(str(rows[1]))  
                    else:
                        image_dict[str(rows[0])] = [str(rows[1])]
    return image_dict


class VARGPT_Edit():

    def __init__(self,):
        model_id = "/mnt/pollux/checkpoints/VARGPT-v1.1-edit"
        prepare_vargpt_qwen2vl_v1_1(model_id)
        self.model = VARGPTQwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,     
            low_cpu_mem_usage=True, 
        ).to(0)

        patching(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = VARGPTQwen2VLProcessor.from_pretrained(model_id)

    def generate(self, img_path, instruction, save_path):

        messages = [
                    {"role": "system", "content": "Your role is to edit the input image <image> only with given instruction <instruction> and keep the content and style of the input image <image> unchanged"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{instruction}"}, # in this image <image>
                            {"type": "image"},
                        ]
                    },
                ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        print (prompt)

        raw_image = Image.open(img_path).resize((512,512))
        inputs = self.processor(images=[raw_image], text=prompt, return_tensors='pt').to(0, torch.float32)

        self.model._IMAGE_GEN_PATH = save_path

        output = self.model.generate(
            **inputs, 
            max_new_tokens=4096, 
            do_sample=False)

        print(self.processor.decode(output[0][:-1], skip_special_tokens=True))



def main():
    args = parse_args()
    image_dict = read_result_csv(args.result_csv_file)
    os.makedirs(args.save_path, exist_ok=True)
    csv_file = f"{args.save_path}/vargpt_edit_meta.csv"

    instruction_gen_model = Llama_Instruction()

    for prompt, paths in tqdm(image_dict.items()):
        instructions = []
        for i in range(args.num_inst):
            instructions.append(instruction_gen_model.generate(prompt))
        print (instructions)

        for path in tqdm(paths):
            for j, instruct in enumerate(instructions):
                image_name = os.path.basename(path)[:4]
                save_path = os.path.join(args.save_path, f"{image_name}_edit_{str(j)}.png")

                model.generate(path, instruct, save_path)
                row = (f"{prompt}", f"{instruct}" ,f"{path}", f"{save_path}")

                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)


if __name__ == '__main__':
    # main()
    
    prompt = 'a cat'
    image_path = '/mnt/pollux/wentian/image_edit/cat.png'
    instruction = [
                "add a colorful rainbow",
                "add a dog and a cat",
                "add a flying saucer",
                "Make it wear a helmet",
    ]
    model = VARGPT_Edit()

    for i, instruct in enumerate(instruction):
        save_path = f'/mnt/pollux/wentian/image_edit/cat_edit_{str(i)}.png'
        model.generate(image_path, instruct, save_path)


    ## CUDA_VISIBLE_DEVICES=7 python vargpt.py

    """
    # Environments
        git clone https://github.com/VARGPT-family/VARGPT-v1.1.git
        conda create -n vargpt python==3.9
        conda activate vargpt
        pip3 install torch torchvision
        pip3 install -r requirements.txt
        pip3 install --upgrade torch torchvision
        pip install timm opencv-python imageio

    # Example

    # Input

        prompt = "A glowing blue sphere with floating metallic rings"

    # Output:
        "add a colorful rainbow."
        "make a robot stand next to the sphere"
        "put a blue planet in the background"
        "make the sphere have a yellow ring."
        "add a dog and a cat"
        "Set a firework in the back"
        "add a flying saucer"
        "place a glowing blue sphere with floating metallic rings in the middle of the
        "Make it wear a helmet"
        "make the sphere a little bit bigger"
    """
