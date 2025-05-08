import os 
import glob
import csv
import json
import argparse
from tqdm import tqdm
import random
from PIL import Image, ImageFile
import torch
import open_clip

def parse_args():
    parser = argparse.ArgumentParser(description="MJHQ 30k Evaluation.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="the number of prompts within a batch",
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args

class MJHQ_Dataset(torch.utils.data.Dataset):
    def __init__(self, mjhq_dir):
        
        image_dict=self.read_result_csv(mjhq_dir)
        self.prompts = list(image_dict.keys())  
        self.paths = list(image_dict.values())

    def read_result_csv(self,csv_path):

        csv_dir_list = glob.glob(os.path.join(csv_path, '*.csv'))
        print (csv_dir_list)

        image_dict = {} #{prompt: image, ...}
        for csv_dir in csv_dir_list:
            with open(csv_dir, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for rows in reader:
                    image_dict[str(rows[0])] = str(rows[1]) 
        print (len(image_dict))
        return image_dict

    def __len__(self):
        return self.prompts.__len__()

    def __getitem__(self, idx):
        caption = self.prompts[idx]
        data = self.paths[idx]

        image = Image.open(data)
        return {"text": caption, "image":image}


def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    return {"text": texts, "image": images}



if __name__ == '__main__':
    args = parse_args()

    clip_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    clip_model = clip_model.to('cuda')
    clip_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

    dataset = MJHQ_Dataset(mjhq_dir=args.data_path)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size,drop_last=False, collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)

    clip_score = 0

    for step, data in enumerate(tqdm(dataloader)):

        caption = data['text']
        images = data['image']

        images = [preprocess(image).unsqueeze(0) for image in images]
        images = torch.cat(images).to('cuda')
        text_tokens = clip_tokenizer(caption).to('cuda')
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        score = (100.0 * (image_features * text_features).sum(axis=-1))
        # print (score.sum().item()/args.batch_size)
        clip_score += score.sum().item()


    print("===>Clip Score:" + str(clip_score/len(dataset)))














