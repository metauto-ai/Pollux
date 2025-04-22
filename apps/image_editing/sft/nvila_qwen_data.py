import json
import os
import shutil
import argparse
import csv

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Test Time Scaling Up', add_help=False)
    parser.add_argument('--result_csv_file', type=str,help='path to <prompt path> csv')
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

class NvilaVerifier():
    def __init__(self):
        model_name = "/mnt/pollux/checkpoints/NVILA-Lite-2B-Verifier"
        cache_dir = '/mnt/pollux/wentian/'
        print("loading NVILA model")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.yes_id = self.model.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id = self.model.tokenizer.encode("no", add_special_tokens=False)[0]
        print("loading NVILA finished")

    def nvila_compare(
        self, 
        image_1, 
        image_2,
        prompt='a cyberpunk cat with a neon sign that says "Sana"',
    ):

        prompt = f"""You are an AI assistant specializing in image analysis and ranking. Your task is to analyze and compare image based on how well they match the given prompt.
    The given prompt is:{prompt}. Please consider the prompt and the image to make a decision and response directly with 'yes' or 'no'.
    """

        r1, scores1 = self.model.generate_content([image_1, prompt])

        r2, scores2 = self.model.generate_content([image_2, prompt])

        return r1, scores1, r2, scores2

    def scoring (self, batch):
        batch_1 = batch['img_model_1']
        batch_2 = batch['img_model_2']
        batch_prompt = batch['caption']

        r1, scores1, r2, scores2 = self.nvila_compare(batch_1, batch_2, batch_prompt)

        if r1 == r2:
            if r1 == "yes":
                # pick the one with higher score for yes
                if scores1[0][0, self.yes_id] > scores2[0][0, self.yes_id]:
                    score=1
                else:
                    score=0
            else:
                # pick the one with less score for no
                if scores1[0][0, self.no_id] < scores2[0][0, self.no_id]:
                    score=1
                else:
                    score=0
        else:
            if r1 == "yes":
                score=1
            else:
                score=0
        return score


class MultimodalAsJudge:
    def __init__(self):
        print("loading Qwen2.5-VL model")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/mnt/pollux/checkpoints/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
        ).cuda()
        min_pixels = 256 * 28 * 28
        max_pixels = 1024 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            "/mnt/pollux/checkpoints/Qwen2.5-VL-7B-Instruct",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        print("loading Qwen2.5-VL finished")

        self.system_prompt = """
        Role:
        You are an expert AI image quality analyst. Your task is to critically evaluate AI-generated images based on their input prompts and provide constructive feedback. For each image-prompt pairs, you have two tasks: \n
        1. Identify Issues: Point out specific problems in the image, including prompt misinterpretation (ambiguous terms, missing details, conflicting instructions), visual artifacts (distortions, anomalies, or rendering issues), composition problems (perspective, layout, or object relationships), style inconsistencies, anatomical/physiological inaccuracies (for living subjects), lighting/texture irregularities. and overall aesthetic appeal\n
        2. Provide Improvement Suggestions: provide actionable fixes (e.g., object adjustments, background adjustments, detail modification), prompt engineering recommendations, style and modifier suggestions.\n
        The output need to be precise, technical while remaining constructive. 
        """

    def generate_image_caption(self, messages):

        # Preparation for inference
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def pil_to_base64(self, image_path):
        image = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


    def scoring(self, img_path, prompt):

        image = self.pil_to_base64(img_path)

        messages = [[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f'data:image;base64,{image}',
                                "min_pixels": 50176,
                                "max_pixels": 50176,
                            },
                            {
                                "type": "text",
                                "text": f'The prompt is {prompt}.',
                            },
                        ],
                    },
                ]]
            
        results = self.generate_image_caption(messages)
        return results



def compare_and_get_loser(img1_path, img2_path, prompt, model):
    """
    Use specific test-time scaling up model to find a worse image from two images with same prompt
    Args:
        img1_path (str): the path of image 1
        img2_path (str): the path of image 2
        prompt (str): the prompt of image 1 and image 2
        model: specific test-time scaling up model
    Return: 
        image path with a lower score < 0.5
    """
    image_1 = Image.open(img1_path).convert("RGB")
    image_2 = Image.open(img2_path).convert("RGB")

    batch = {
        "caption": prompt, 
        "img_model_1": image_1,
        "img_model_2": image_2,       
    }
    res = model.scoring(batch)
    if res > 0.5:
        loser=img2_path
    else:
        loser=img1_path
    return loser


def find_last_place(results, prompt, model):
    """
    find the worst image from N result candidates according to the score.
    Args:
        results (list): N image paths
        prompt  (str): the prompt of results
        model: specific test-time scaling up model
    Return: 
        the path of the worst image
    """
    worst = results[0]
    for candidate in results[1:]:
        worst = compare_and_get_loser(worst, candidate, prompt, model)
    return worst

def main():
    args = parse_args()
    image_dict = read_result_csv(args.result_csv_file)
    csv_file = f"{args.save_path}/Nvila_Qwen_prompt_reason_worst_meta.csv"

    judge = NvilaVerifier()
    interrogator = MultimodalAsJudge()

    for prompt, paths in tqdm(image_dict.items()):
        worst_image_path=find_last_place(paths, prompt=prompt, model=judge)
        reason = interrogator.scoring(worst_image_path, prompt)
        row = (f"{prompt}", f"{reason[0]}", f"{worst_image_path}")
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)

if __name__ == "__main__":
    main()
    """
    # Environments
    pip install qwen-vl-utils
    pip install transformers
    pip install git+https://github.com/bfshi/scaling_on_scales.git
    
    # Example
    
    candidates = [
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-0.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-1.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-2.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-3.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-4.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-5.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-6.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-7.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-8.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-9.png",
    "/mnt/pollux/geneval_bench/generate_tmp/test_time_scaling_up/flux_dev/00055-10.png"
    ]
    prompt = "a photo of a dining table"
    judge = NvilaVerifier()
    interrogator = MultimodalAsJudge()

    worst_image_path = find_last_place(candidates, prompt, judge)
    print (worst_image_path)
    reason = interrogator.scoring(worst_image_path, prompt)
    print (reason[0])

    # Output:

    ### Identification of Issues:

    1. **Prompt Misinterpretation**: The prompt specifies "a photo of a dining table," but the image appears to be a wider view that includes multiple tables and chairs, suggesting it might be more accurately described as a dining area rather than just one table.
       
    2. **Visual Artifacts**: There are no apparent distortions or rendering issues in the image itself; however, the lighting seems slightly uneven, with some areas appearing brighter than others, which could be due to natural light coming through the windows.

    3. **Composition Problems**: The composition is somewhat cluttered because there are multiple tables and chairs visible,
    """
