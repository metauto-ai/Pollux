import os
import glob
import boto3
import json
import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------- AWS --------
client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id='AKIA47CRZU7STC4XUXER',
    aws_secret_access_key='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',
    region_name='us-east-1'
)

# -------- System Prompt --------
system_prompt = """You are tasked with generating image captions that will be used for training diffusion text-to-image models. 
                   Your goal is to create captions that imitate what humans might use as prompts when generating images. 
Guidelines for generating image captions:
1. Please generate a comprehensive, single-paragraph caption that includes every visible element in the image, including any readable text. 
2. Include information about style, mood, lighting, and composition when relevant.
3. Use a mix of concrete and abstract terms.
4. Incorporate artistic references or techniques when appropriate.
5. Ensure the caption is free from imaginary content or hallucinations. 
6. Present all information in one cohesive narrative without using structured lists.
Use the subject as the main focus of your caption and incorporate elements from the style to enhance the description. Combine these elements creatively to produce a unique and engaging caption.
Begin your response with "Image Caption:"
"""
system = [{"text": system_prompt}]


def encode_image(image_path):

    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        image_bytes = buffer.getvalue()
    return image_bytes


def generate_image_caption(image_path):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
                        "source": {
                            "bytes": encode_image(image_path)
                        }
                    }
                }
            ]
        }
    ]

    inf_params = {
        "maxTokens": 200,
        "topP": 0.1,
        "temperature": 1.0
    }

    additionalModelRequestFields = {
        "inferenceConfig": {
            "topK": 20
        }
    }

    model_response = client.converse(
        modelId="us.amazon.nova-lite-v1:0",
        messages=messages,
        system=system,
        inferenceConfig=inf_params,
        additionalModelRequestFields=additionalModelRequestFields
    )

    return model_response


if __name__ == "__main__":

    image_folder = "/jfs/data/**/images"
    image_paths = (
        glob.glob(os.path.join(image_folder, "*.jpg")) +
        glob.glob(os.path.join(image_folder, "*.jpeg")) +
        glob.glob(os.path.join(image_folder, "*.png"))
    )

    max_workers = 12
    need_regen = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(generate_image_caption, img_path): img_path
            for img_path in image_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                response = future.result()
            except Exception as e:
                need_regen.append(path)
                print(f"Erros in handling {path}：{e}")
            else:
 
                print(f"\n[Full Response for {path}]")
                print(json.dumps(response, indent=2, ensure_ascii=False))
                print(f"\n[Response Content Text for {path}]")
                print(response["output"]["message"]["content"][0]["text"])
