import argparse
import datetime
import json 
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm 

from instruction_generation import instruction_generation
from random import choice, shuffle
from step1x_edit import Step1x_Edit
from hidream_edit import HiDream_Edit

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='step1x_edit', help='Image editing model [step1x_edit/hidream_edit]')
    parser.add_argument('--task', type=str, required=True, default='add', help='Image editing task [add/remove/background]')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output image directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the JSON file containing image names and prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of diffusion steps')
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG guidance strength')
    parser.add_argument('--size_level', default=512, type=int)
    # parser.add_argument('--offload', action='store_true', help='Use offload for large models')
    # parser.add_argument('--quantized', action='store_true', help='Use fp8 model weights')
    parser.add_argument('--num_inst', type=int, default=1, help='num instruction per prompt')
    parser.add_argument('--num_image', type=int, default=1, help='num image generation per instruction')
    args = parser.parse_args()

    assert os.path.exists(args.csv_path), f"Input directory {args.csv_path} does not exist."
    assert os.path.exists(args.json_path), f"JSON file {args.json_path} does not exist."

    # args.output_dir = args.output_dir.rstrip('/') + ('-offload' if args.offload else "") + ('-quantized' if args.quantized else "") + f"-{args.size_level}"
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = f"{args.output_dir}/{args.model}_meta.csv"

    # 1. read prompt-image dict
    image_dict = read_result_csv(args.csv_path)

    # 2. initial instruction generation model and image generation model
    instruction_gen_model = instruction_generation(args.task)
    if args.model == 'step1x_edit':
        image_edit = Step1x_Edit()
    elif args.model == 'hidream_edit':
        image_edit = HiDream_Edit()
    time_list = []

    # 3. generate edit images
    for prompt, paths in image_dict.items():

        # 3.1 generate instructions
        instructions = []
        if args.task in ['add', 'remove']:
            for i in range(args.num_inst):
                instruct = instruction_gen_model.generate(prompt, args.task)
                instructions.append(instruct)

        elif args.task in ['background', 'text', 'style']:
            path_ = choice(paths)
            instruct = instruction_gen_model.generate(img_path=path_, 
                                                      prompt=prompt, 
                                                      num_inst=args.num_inst, 
                                                      task=args.task
                                                    )
            instructions.extend(instruct)
        elif args.task == 'lighting':
            path_ = choice(paths)
            instruct = instruction_gen_model.generate(img_path=path_, 
                                                      prompt=None, 
                                                      num_inst=args.num_inst, 
                                                      task=args.task
                                                    )
            instructions.extend(instruct)

        # 3.2 generate images
        for path in tqdm(paths):
            for j, instruct in enumerate(instructions):
                image_name = os.path.basename(path)[:4]
                start_time = time.time()
                for k in range(args.num_image):
                    output_path = os.path.join(args.output_dir, f"{image_name}_{args.task}_{str(j)}_num_{str(k)}.png")
                    image = image_edit.generate(
                        instruct = instruct,
                        image_path =path, 
                        num_steps=args.num_steps,
                        cfg_guidance=args.cfg_guidance,
                        seed=args.seed)
                    
                    print(f"Time taken: {time.time() - start_time:.2f} seconds")
                    time_list.append(time.time() - start_time)

                    image.save(
                        os.path.join(output_path), lossless=True
                    )

                    row = (f"{prompt}", f"{instruct}" ,f"{path}", f"{output_path}")

                    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)


    print(f'average time for {args.output_dir}: ', sum(time_list[1:]) / len(time_list[1:]))


if __name__ == "__main__":
    main()
