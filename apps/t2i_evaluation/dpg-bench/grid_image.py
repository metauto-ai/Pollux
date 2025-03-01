import os
import csv
import argparse
import numpy as np
from tqdm import tqdm
import glob
import torch
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench Grid Images.")
    parser.add_argument(
        "--result_csv_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt_map_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--grid_save_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def read_result_csv(csv_path, prompt_map_path):
    
    prompt_map = {} # {prompt: filename, ...}
    with open(prompt_map_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for rows in reader:
            prompt_map[str(rows[0])] = str(rows[1]) 
    print (len(prompt_map))

    csv_dir_list = glob.glob(csv_path+'*.csv')
    print (csv_dir_list)

    image_dict = {} # {prompt_filename: [image1, image2, image3], ...}
    for csv_dir in csv_dir_list:
        with open(csv_dir, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for rows in reader:
                name = prompt_map[rows[0]]
                if name in image_dict:
                    image_dict[name].append(str(rows[1]))  
                else:
                    image_dict[name] = [str(rows[1])]
    print (len(image_dict))
    return image_dict


def check(image_dict):

    num = len(image_dict)

    error_line = 0
    for name, paths in image_dict.items():
        if len(paths) != 4:
            print (name)
            error_line +=1

    state = 1 if num == 1065 and error_line == 0 else 0
    print (f'There are {num} (4260/4) prompt samples with {error_line} vacancy!')
    return state



def grid_image(csv_path, prompt_map_path, save_path):

    """
    csv_path :  the dir of csv file storing the prompts and their 4 images.
    prompt_map_path: the dir of csv file storing the prompts and their filenames.
    save_path:  the path to save grid images.
    """
    image_dict = read_result_csv(csv_path, prompt_map_path)
    os.makedirs(save_path, exist_ok=True)
    state = check(image_dict)

    if state ==1:
        for name, paths in tqdm(image_dict.items()):
            images = []
            for path in paths:
                img = Image.open(path)
                img = ToTensor()(img)
                images.append(img)

            images = torch.stack(images)
            grid = make_grid(images, nrow=2)
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid.save(os.path.join(save_path, f'{name}.png'))
    else:
        print ('please check your input samples')
        return

if __name__ == "__main__":
    args = parse_args()
    grid_image(args.result_csv_path, args.prompt_map_path, args.grid_save_path)





