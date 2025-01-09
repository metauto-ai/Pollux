import boto3
import json
import base64
from PIL import Image
import io

# Add AWS credentials
client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id='AKIA47CRZU7STC4XUXER',
    aws_secret_access_key='w4B1K9YL32rwzuZ0MAQVukS/zBjAiFBRjgEenEH+',
    region_name='us-east-1'  # specify your region
)

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
system = [{ "text": system_prompt }]

def encode_image(image_path):
    # Open and convert image to bytes
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Save as JPEG in memory with high quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)  # Reset buffer position to start
        image_bytes = buffer.getvalue()
        # Return raw bytes instead of base64 encoding
        return image_bytes

# Update messages structure
messages = [
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": encode_image("/Users/aj/Downloads/image.jpg")
                    }
                }
            }
        ]
    }
]

inf_params = {"maxTokens": 200, "topP": 0.1, "temperature": 1.0}

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

print("\n[Full Response]")
print(json.dumps(model_response, indent=2))

print("\n[Response Content Text]")
print(model_response["output"]["message"]["content"][0]["text"])
