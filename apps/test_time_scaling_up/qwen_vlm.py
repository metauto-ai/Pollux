from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64
import re


class MultimodalAsJudge:
    def __init__(self):
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
        self.system_prompt = """
        Role:
        You are an expert vision-language model designed to evaluate two images based on their alignment with a given text prompt and their visual quality.

        Task Description:
        Evaluate two images based on their alignment with the provided prompt and their visual quality. Assign a float score between 0 and 1, where:

        Score close to 1 — Image 1 is significantly better aligned with the prompt and has superior visual quality.
        Score close to 0 — Image 2 is significantly better aligned with the prompt and has superior visual quality.
        
        Input Format:
        You will receive:
        Two images: Image 1 and Image 2
        A text prompt describing the desired visual content

        Evaluation Criteria:

        Prompt Alignment:
        Assess how accurately the content, composition, and key visual elements match the provided prompt.
        Prioritize fidelity to the described scene, objects, and attributes.
        
        Visual Quality:
        Evaluate clarity, sharpness, and detail.
        Penalize artifacts, distortions, and unrealistic visual elements.
        
        Prompt Alignment is important than Visual Quality. If both images are equally aligned with the prompt, prioritize Visual Quality.

        Output Format:
        Return a single float score in the format: Score: <value>
        Avoid additional comments or explanations unless explicitly requested.
        
        Example Prompt:
        "A cat sitting on a wooden table in a cozy room with warm lighting."

        Example Output:
        If Image 1 is significantly better: Score: 0.9
        If Image 2 is significantly better: Score: 0.1

        Important Notes:

        The score should only reflect relative comparison — not absolute quality scores.
        Maintain consistency in scoring even if both images are poor-quality or fail to align with the prompt.
    

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

    def pil_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def filter_caption(self, caption):
        match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", caption)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Score not found in response.")

    def scoring(self, batch):
        assert "caption" in batch, "The batch must contain a 'caption' key."
        assert "img_model_1" in batch, "The batch must contain an 'image_1' key."
        assert "img_model_2" in batch, "The batch must contain an 'image_2' key."
        batch["img_model_1"] = [
            self.pil_to_base64(image) for image in batch["img_model_1"]
        ]
        batch["img_model_2"] = [
            self.pil_to_base64(image) for image in batch["img_model_2"]
        ]
        messages = []
        for idx in range(len(batch["caption"])):
            messages.append(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f'data:image;base64,{batch["img_model_1"][idx]}',
                                "min_pixels": 50176,
                                "max_pixels": 50176,
                            },
                            {
                                "type": "image",
                                "image": f'data:image;base64,{batch["img_model_2"][idx]}',
                                "min_pixels": 50176,
                                "max_pixels": 50176,
                            },
                            {
                                "type": "text",
                                "text": f'The prompt is {batch["caption"][idx]}.',
                            },
                        ],
                    },
                ]
            )
        results = self.generate_image_caption(messages)
        filtered_results = [self.filter_caption(res) for res in results]
        return filtered_results


if __name__ == "__main__":
    import requests
    from PIL import Image
    from io import BytesIO

    url = "https://ayon.blob.core.windows.net/pexelimages/22863019.jpg"
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image with PIL
    image_1 = Image.open(BytesIO(response.content))

    url = "https://ayon.blob.core.windows.net/pexelimages/18853330.jpg"
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image with PIL
    image_2 = Image.open(BytesIO(response.content))

    # prompt = "A close-up portrait of a charming orange and white cat with striking green eyes and a pink nose, lounging comfortably in front of a lush, vibrant bouquet of lavender flowers. The cat's fur is a mix of soft, creamy white and warm, rich orange, with a tuft of white fur on its chest. The lavender flowers, with their delicate purple petals and green leaves, are arranged in a tall, slender vase, adding a touch of natural elegance to the scene. The lighting is soft and warm, casting gentle shadows and highlighting the cat's expressive features and the intricate details of the flowers. The composition is intimate and cozy, capturing the serene and peaceful atmosphere of a moment spent with a beloved pet."
    prompt = "A meticulously arranged table setting features a three-tiered tiered tray with a top layer adorned with fresh rosemary leaves, a middle layer showcasing neatly sliced lemons, and a bottom layer with a variety of small, round, yellow biscuits. Adjacent to the tray, a wooden bowl filled with dark, aromatic spices sits on a wooden stand, while another bowl contains small, bright red berries. Behind the tray, an assortment of bottles with labels such as `S` and `Gin` are arranged in a row, creating a warm and inviting atmosphere with soft, ambient lighting."

    batch = {"caption": [prompt], "img_model_1": [image_1], "img_model_2": [image_2]}
    mm_judge = MultimodalAsJudge()
    res = mm_judge.scoring(batch)
    print(res)
