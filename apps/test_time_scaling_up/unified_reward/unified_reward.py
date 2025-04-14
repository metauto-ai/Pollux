from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import itertools
from PIL import Image
import requests
import copy
import torch
import warnings
import os
from datasets import load_dataset, load_from_disk
import tqdm
import json




pretrained = "/mnt/pollux/checkpoints/UnifiedReward-7b-v1.5"
warnings.filterwarnings("ignore")


model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
conv_template = "qwen_1_5"
tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                        pretrained, 
                                                        None, 
                                                        model_name, 
                                                        device_map=device_map,
                                                        attn_implementation=None,
)
model.eval()


def data_process(image_1, image_2):

    image1 = image_1.resize((512, 512))
    image2 = image_2.resize((512, 512))
    image_sizes = [image1.size, image2.size]
    image_tensor = process_images([image1, image2], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    return image_tensor, image_sizes


def judge(image_1, image_2, prompt):

    image_tensor, image_sizes = data_process(image_1, image_2)
    question = f'<image>\n <image>\nYou are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Image 1 is better than Image 2.\", \"Image 2 is better than Image 1.\" and \"Both images are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]'
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]

def unified_reward_judge_func(batch_1, batch_2, batch_prompt):

    assert len(batch_1)==len(batch_2)
    assert len(batch_1)==len(batch_prompt)

    score = 0
    answer = 'Image 1 is better than Image 2'

    num_all = len(batch_1)
    for i in range(len(batch_1)):
        text_output = judge(batch_1[i], batch_2[i], batch_prompt[i])

        if answer in text_output:
            score += 1
            
    score = score / num_all

    return score

if __name__ == '__main__':

    image1 = Image.open('/home/ubuntu/wentian/test_time_scaling_up/UnifiedReward/output-0.jpg')
    image2 = Image.open('/home/ubuntu/wentian/test_time_scaling_up/UnifiedReward/output-1.jpg')

    image_batch_1 = [image1, image1, image1]
    image_batch_2 = [image2, image2, image2]

    prompt_batch = [
        "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns",
        "A beautiful asian woman in traditional clothing with golden hairpin and blue eyes, wearing a red kimono with dragon patterns"
        ]


    score = unified_reward_judge_func(image_batch_1, image_batch_2, prompt_batch)
    print (score)

