import torch
from PIL import Image
from pathlib import Path
from img_filter import ImgFilterArgs, build_filter

# Assuming the previous code is already imported and available

# Define the images and their corresponding prompts
to_test_images = [
    (
        "/jfs/jinjie/code/Pollux/apps/preprocessing/LAION-5B-WatermarkDetection/images/clear_example.png",
        "a football player in red raises his hand",
    ),
    (
        "/jfs/jinjie/code/Pollux/apps/preprocessing/LAION-5B-WatermarkDetection/images/watermark_example.png",
        "a very large house",
        # "a family walking on the street",
    ),
]

watermark_filter_args = ImgFilterArgs(
    model_name="WaterMarkFilter",
    pretrained_model_name_or_path="/jfs/checkpoints/data_preprocessing/watermark_model_v1.pt",
)

# Define the clip filter args with the model path
clip_filter_args = ImgFilterArgs(
    model_name="CLIPFilter",
    pretrained_model_name_or_path="openai/clip-vit-base-patch16",
)


# Function to load the image
def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


# Test the filters
def test_filters():
    for image_path, prompt in to_test_images:
        print(f"Testing image: {image_path}")

        # Load the image
        image = load_image(image_path)

        # Test WaterMarkFilter
        print("\nTesting WaterMarkFilter...")
        watermark_filter = build_filter(watermark_filter_args)
        watermark_score = watermark_filter.predict(image)
        print(f"Watermark score (clear probability): {watermark_score:.4f}")

        # Test CLIPFilter
        print("\nTesting CLIPFilter...")
        clip_filter = build_filter(clip_filter_args)
        clip_score = clip_filter.predict(image, prompt=prompt)
        print(f"CLIP score (0-100, higher is better): {clip_score:.4f}")


# Run the test
test_filters()
