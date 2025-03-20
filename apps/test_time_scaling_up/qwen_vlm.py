from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64


class MultimodalAsJudge:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/mnt/pollux/checkpoints/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
        ).cuda()
        self.processor = AutoProcessor.from_pretrained(
            "/mnt/pollux/checkpoints/Qwen2.5-VL-7B-Instruct"
        )
        self.system_prompt = """
        Role:
        
        You are an expert vision-language model trained to evaluate image quality and alignment with a given text prompt. Your goal is to assess two images and assign a score that reflects their overall alignment with the prompt and visual quality.

        Evaluation Criteria:

        Prompt Alignment:

        Assess how accurately the content, composition, and key visual elements match the provided prompt.
        Prioritize fidelity to the described scene, objects, and attributes.
        Visual Quality:

        Evaluate clarity, sharpness, and detail.
        Penalize artifacts, distortions, and unrealistic visual elements.
        Scoring Guidelines:

        Assign a float score between 0 and 1:
        1.0 — Perfect alignment with the prompt and flawless visual quality.
        0.8 - 0.9 — Strong alignment with minimal imperfections.
        0.6 - 0.7 — Partial alignment with notable quality issues or prompt deviation.
        0.3 - 0.5 — Weak alignment or significant visual defects.
        0.0 - 0.2 — Severe distortion, irrelevant content, or complete mismatch.
        Output Format:

        Return only the score in the format: Score: <value>
        Do not provide explanations unless requested.
        Example Prompt:
        "A cat sitting on a wooden table in a cozy room with warm lighting."

        Example Image Evaluations:

        Image A: Clear, sharp, and faithfully represents the scene — Score: 0.9
        Image B: Blurry with distorted lighting and missing key elements — Score: 0.4
        Instructions:

        Maintain consistency in scoring.
        Focus strictly on visual alignment and quality without external bias.
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
                            },
                            {
                                "type": "image",
                                "image": f'data:image;base64,{batch["img_model_2"][idx]}',
                            },
                            {
                                "type": "text",
                                "text": f'The caption is {batch["caption"][idx]}.',
                            },
                        ],
                    },
                ]
            )
        return self.generate_image_caption(messages)


if __name__ == "__main__":
    import requests
    from PIL import Image
    from io import BytesIO

    url = "https://ayon.blob.core.windows.net/flickr-images/13586111764.jpg"
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image with PIL
    image_1 = Image.open(BytesIO(response.content))

    url = "https://ayon.blob.core.windows.net/flickr-images/13587636944.jpg"
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the image with PIL
    image_2 = Image.open(BytesIO(response.content))

    prompt = "A serene harbor scene unfolds with several boats gently bobbing on the calm waters, their reflections shimmering on the surface. In the foreground, a small white boat with a black outboard motor is anchored, while a larger boat with a blue cover and a wooden deck is moored nearby. The backdrop features a picturesque row of vibrant, multicolored buildings with terracotta roofs, standing against a backdrop of a hill crowned with a medieval castle. The sky is clear and bright, casting a warm glow over the scene, and the overall mood is tranquil and idyllic, reminiscent of a classic Mediterranean coastal town."

    batch = {"caption": [prompt], "img_model_1": [image_1], "img_model_2": [image_2]}
    mm_judge = MultimodalAsJudge()
    res = mm_judge.scoring(batch)
    print(res)
