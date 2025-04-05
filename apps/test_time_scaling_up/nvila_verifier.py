import json
import os
import os.path as osp
import shutil

import numpy as np
import PIL.Image
from tqdm import tqdm
from transformers import AutoModel

class NvilaVerifier():
    def __init__(self):
        model_name = "/mnt/pollux/checkpoints/NVILA-Lite-2B-Verifier"
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
        batch_1 = batch['img_model_2']
        batch_prompt = batch['caption']
        assert len(batch_1)==len(batch_2)
        assert len(batch_1)==len(batch_prompt)

        score = []

        for i in range(len(batch_1)):
            r1, scores1, r2, scores2 = self.nvila_compare(batch_1[i], batch_2[i], batch_prompt[i])

            if r1 == r2:
                if r1 == "yes":
                    # pick the one with higher score for yes
                    if scores1[0][0, self.yes_id] > scores2[0][0, self.yes_id]:
                        score.append(1)
                    else:
                        score.append(0)
                else:
                    # pick the one with less score for no
                    if scores1[0][0, self.no_id] < scores2[0][0, self.no_id]:
                        score.append(1)
                    else:
                        score.append(0)
            else:
                if r1 == "yes":
                    score.append(1)
                else:
                    score.append(0)
        return score


