import requests
import math
from PIL import Image
from io import BytesIO

sample = {
    "source_id": "679011b2e28d45828f71b3f8",
    "media_path": "https://ayon.blob.core.windows.net/pexelimages/10954867.jpg",
    "partition_key": 7562,
    "caption": "An empty asphalt road stretches into the distance, flanked by a dense …",
    "width": 5457,
    "height": 3635,
    "source": "pexel_images",
}

PATCH_SIZE = 2
VAE_COMPRESS_RATIO = 8


def crop_to_fit_model(image, basic_pixel_patch_size):
    width = image.width
    height = image.height
    crop_width = (width // basic_pixel_patch_size) * basic_pixel_patch_size
    crop_height = (height // basic_pixel_patch_size) * basic_pixel_patch_size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image


def transform(
    sample, basic_pixel_patch_size, current_resolution=768, cond_resolution=256
):
    response = requests.get(sample["media_path"], timeout=10)
    ori_image = Image.open(BytesIO(response.content)).convert("RGB")
    image = ori_image.copy()
    # Compute scaling factor to achieve 768x768 area
    scale_factor = math.sqrt(
        (current_resolution * current_resolution) / (image.width * image.height)
    )
    cond_scale_factor = cond_resolution / current_resolution

    new_width = round(image.width * scale_factor)
    new_height = round(image.height * scale_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    image = crop_to_fit_model(image, basic_pixel_patch_size)

    cond_width = round(new_width * cond_scale_factor)
    cond_height = round(new_height * cond_scale_factor)
    cond_image = image.resize((cond_width, cond_height), Image.LANCZOS)
    cond_image = crop_to_fit_model(cond_image, basic_pixel_patch_size)
    return ori_image, image, cond_image


def main():
    ori_image, image, cond_image = transform(
        sample,
        basic_pixel_patch_size=PATCH_SIZE * VAE_COMPRESS_RATIO,
        current_resolution=768,
        cond_resolution=256,
    )
    ori_image.save("originalImage.jpg")
    print("Original Image saved with shape: ", ori_image.size)
    image.save("imageforGenModel.jpg")
    print("Image for Gen Model saved with shape: ", image.size)
    cond_image.save("imageforPlanModel.jpg")
    print("Image for Plan Model saved with shape: ", cond_image.size)
    assert (
        image.width >= 768 or image.height >= 768
    ), "Resized dimensions should be at least 768 in one direction."
    assert (
        image.width % 16 == 0 and image.height % 16 == 0
    ), "Cropped dimensions must be divisible by 16."
    assert abs((image.width * image.height) - (768 * 768)) <= (
        2 * 16 * 768 - 16 * 16
    ), "Cropped area should be close to 589,824 pixels."
    assert abs((cond_image.width * cond_image.height) - (256 * 256)) <= (
        2 * 16 * 256 - 16 * 16
    ), "Cropped area should be close to 589,824 pixels."
    print("All tests passed successfully.")


if __name__ == "__main__":
    main()
