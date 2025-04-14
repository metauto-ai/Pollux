import os
import csv
import glob
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Test Time Scaling Up', add_help=False)

    parser.add_argument('--result_csv_file', type=str,help='path to <prompt path> csv')
    parser.add_argument('--top_k',default = 4,type=int,help='top k to save')
    parser.add_argument('--model',default = 'nvila_verifier',type=str,help='which model for test time scaling up')
    parser.add_argument('--save_path', metavar='DIR',help='path to save <prompt path> csv')
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
        print (csv_path)
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for rows in reader:
                if (rows[0]) in image_dict:
                    image_dict[str(rows[0])].append(str(rows[1]))  
                else:
                    image_dict[str(rows[0])] = [str(rows[1])]

    elif os.path.isdir(csv_path):
        csv_dir_list = glob.glob(csv_path+'*.csv')
        print (csv_dir_list)
        for csv_dir in csv_dir_list:
            with open(csv_dir, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for rows in reader:
                    if (rows[0]) in image_dict:
                        image_dict[str(rows[0])].append(str(rows[1]))  
                    else:
                        image_dict[str(rows[0])] = [str(rows[1])]
    print (len(image_dict))
    return image_dict


def compare(img1_path, img2_path, prompt, model):
    """
    Use specific test-time scaling up model to find a better image from two images with same prompt
    Args:
        img1_path (str): the path of image 1
        img2_path (str): the path of image 2
        prompt (str): the prompt of image 1 and image 2
        model: specific test-time scaling up model
    Return: 
        image path with a higher score > 0.5
    """
    image_1 = Image.open(img1_path).convert("RGB")
    image_2 = Image.open(img2_path).convert("RGB")

    batch = {
        "caption": [prompt], 
        "img_model_1": [image_1],
        "img_model_2": [image_2],       
    }
    res = model.scoring(batch)
    if res[0] > 0.5:
        return img1_path
    else:
        return img2_path

def tournament_select(results, prompt, k=4, model):
    """
    Tournament selects top k images from N result candidates according to the score.
    Args:
        results (list): N image paths
        prompt  (str): the prompt of results
        k (int) : the top number of samples from results
        model: specific test-time scaling up model
    Return: 
        a list contains top k image paths
    """
    winners = results.copy()
    top_k = []

    for _ in range(k):
        while len(winners) > 1:
            next_winners = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    next_winners.append(compare(winners[i], winners[i + 1], prompt=prompt, model=model))
                else:
                    next_winners.append(winners[i]) 
            winners = next_winners
        top_k.append(winners[0])
        results = [x for x in results if x != winners[0]]
        winners = results.copy()

    return top_k


def main():
    args = parse_args()
    image_dict = read_result_csv(args.result_csv_file)
    csv_file = f"{args.save_path}/{args.model}_top{str(args.top_k)}_meta.csv"

    if args.model == 'unified_reward':
        from unified_reward.unified_reward import UnifiedReward
        judge = UnifiedReward()

    elif args.model == 'qwen':
        from qwen_vlm import MultimodalAsJudge
        judge = MultimodalAsJudge()
        
    elif args.model == 'nvila':
        from nvila_verifier import NvilaVerifier
        judge = NvilaVerifier()

    for prompt, paths in tqdm(image_dict.items()):
        image_dict_topk=tournament_select(paths, prompt=prompt, k=args.top_k, model=judge)
        for path in image_dict_topk:
            row = (f"{prompt}", f"{path}")
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)


if __name__ == '__main__':
    main()


