import random 
from random import choice, shuffle
import torch
import transformers

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64
import re
from PIL import Image
import os

class Llama_Instruction():
    def __init__(self,):

        model_id = "/mnt/pollux/checkpoints/Llama-3.1-8B"

        self.pipeline = transformers.pipeline(
            "text-generation", 
            model=model_id, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device_map="auto"
        )

        # model_id = "meta-llama/Llama-3.3-70B-Instruct"
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model_id,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device_map="auto",
        # )

    def get_content_instruction(self, new_prompt):
        few_shot_examples = [
            """
            [INST]User: "Beautiful cat with mojito sitting in a cafe on the street"[/INST] 
            Assistant: {"edit": "include a hat", "edited object": "hat", "output": "Beautiful cat wearing a hat with mojito sitting in a cafe on the street"}""",
            """
            [INST]User: "robot playing chess at home."[/INST] 
            Assistant: {"edit": "add a cheerful smiling face.", "edited object": "robot", "output": "robot playing chess at home with a cheerful smiling face."} """,
            """
            [INST]User: "A cute creature sits at the beach."[/INST] 
            Assistant: {"edit": "set a dog besides the creature", "edited object": "dog", "output": "A cute creature and a dog sit at the beach."} """,
            """
            [INST]User: "Superhero on the street in sunny day working on his tablet."[/INST] 
            Assistant: {"edit": "put a vintage tie on the superhero.", "edited object": "tie", "output": "Superhero with a vintage tie on the street in sunny day working on his tablet."} """,
            """
            [INST]User: "Picture clouds, birds, the wind, foliage, rainbow, hill, art, pair, guy"[/INST] 
            Assistant: {"edit": "together with a dog on the left", "edited object": "dog", "output": "Picture clouds, birds, the wind, foliage, rainbow, hill, art, pair, guy, dog on the left"} """,
            """
            [INST]User: "horse on a red Boat Near Mountains During Golden Hour"[/INST] 
            Assistant: {"edit": "give the horse sunglassess", "edited object": "sunglassess", "output": "horse with sunglassess on a red Boat Near Mountains During Golden Hour"} """,
            """
            [INST]User: "An animal family on studio background."[/INST] 
            Assistant: {"edit": "make them hold a teddy bear.", "edited object": "teddy bear", "output": "An animal family holding a teddy bear on studio background."} """,
            """
            [INST]User: "Baked Salmon With Bell Peppers"[/INST] 
            Assistant: {"edit": "insert kale pesto to the dish", "edited object": "kale pesto", "output": "Baked Salmon With Kale Pesto And Bell Peppers"} """,
            """
            [INST]User: "An airplaine is flying in the sky in rainy day."[/INST] 
            Assistant: {"edit": "add flowers in the windows", "edited object": "flowers", "output": "An airplaine with flowers in the windows is flying in the sky in rainy day."} """,
            """
            [INST]User: "photo of mountains and trees"[/INST] 
            Assistant: {"edit": "position a castle between the trees", "edited object": "castle", "output": "photo of mountains, trees and castle between the trees"} """,
            """
            [INST]User: "Attic Bedroom With Large Ceilings"[/INST] 
            Assistant: {"edit": "decorate the room with beautiful chandeliers", "edited object": "chandeliers", "output": "Attic Bedroom With Beautiful Chandeliers on Large Ceilings"} """,
            """
            [INST]User: "Wedding rings and yellow flower on a red background"[/INST]  
            Assistant: {"edit": "place a bird by the yellow flower", "edited object": "bird", "output": "Wedding rings, a bird, and yellow flower on a red background"} """,
            """
            [INST]User: "Tree Near the lake in the morning"[/INST] 
            Assistant: {"edit": "Give it autumn leaves on top", "edited object": "leaves", "output": "Tree with autumn leaves on top Near the lake in the morning"} """,
            """
            [INST]User: "robot and alien sitting on hanging bridge at daytime"[/INST]  
            Assistant: {"edit": "make them hold three books.", "edited object": "threebooks", "output": "robot and alien holding three books while sitting on hanging bridge at daytime"} """,
            """
            [INST]User: "Skogafoss waterfall in the south of Iceland"[/INST]  
            Assistant: {"edit": "Set a colorful rainbow in the backhground!", "edited object": "rainbow", "output": "Skogafoss waterfall with a colorful rainbow in the south of Iceland"} """,
            """
            [INST]User: "Polar Bear with rubber gloves pushing shopping carts"[/INST]  
            Assistant: {"edit": "Make it wear a coat", "edited object": "coat", "output": "Polar Bear with a coat pushing shopping carts"}
            """
            ]
        optional_verbs = choice(["include", "place", "position", "set", "incorporate", "alongside", "give", "put", "insert", "together with", "with", "integrate", "have", "append", "add", "include"])
        
        # system message 
        system_message = f"""
        <<SYS>>You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is 
            JSON with the following string fields: 'edit', 'edited object','output'. 
            Here is the latest conversation between Assistant and User.
        <</SYS>>
        """

        # introduction
        intro_message = f"""
        [INST]User: Hi, My job to take a given caption ('input') and to output the following: an instruction for {optional_verbs} an object to the image ('edit'), the object to {optional_verbs} ('edited object'), and the caption with the object ('output'). Please help me do it.
            I will give you the 'input', and you will help. When you reply, use the following format: {{"edit": '<instruction>', 'edited object': '<object>', 'output': '<caption>'}}
        [/INST]
        Assistant: Sure, I'd be happy to help! Please provide the actual input caption you'd like me to read and I'll assist you with writing an instruction to {optional_verbs} an object to the image, writing the added object and writing the caption with the object."""  
        
        # shuffling #
        random.seed(torch.randint(1 << 32, ()).item())
        shuffle(few_shot_examples)
        few_shot_examples = few_shot_examples[:int(len(few_shot_examples) * 0.6)]
        prompt = system_message + intro_message + "".join(few_shot_examples)        # add the test prompt
        prompt = prompt + f"[INST]User: {new_prompt}[/INST]"

        return prompt
    def del_content_instruction(self, new_prompt):
        few_shot_examples = [
            """
            [INST]User: "Beautiful cat with mojito sitting in a cafe on the street"[/INST] 
            Assistant: {"edit": "omit the mojito", "edited object": "mojito", "output": "Beautiful cat sitting in a cafe on the street"}""",
            """
            [INST]User: "robot playing chess at home."[/INST] 
            Assistant: {"edit": "take out the chess.", "edited object": "chess", "output": "robot at home."} """,
            """
            [INST]User: "A cute creature sits at the beach."[/INST] 
            Assistant: {"edit": "erase the cute creature", "edited object": "cute creature", "output": "beach"} """,
            """
            [INST]User: "Superhero on the street in sunny day working on his tablet."[/INST] 
            Assistant: {"edit": "drop the tablet", "edited object": "tablet", "output": "Superhero on the street in sunny day"} """,
            """
            [INST]User: "Picture clouds, birds, the wind, foliage, rainbow, hill, art, pair, guy"[/INST] 
            Assistant: {"edit": "remove rainbow and foliage", "edited object": "rainbow and foliage", "output": "Picture clouds, birds, the wind, hill, art, pair, guy"} """,
            """
            [INST]User: "horse on a red Boat Near Mountains During Golden Hour"[/INST] 
            Assistant: {"edit": "delete the horse", "edited object": "horse", "output": "a red Boat Near Mountains During Golden Hour"} """,
            """
            [INST]User: "An animal family on studio background."[/INST] 
            Assistant: {"edit": "exclude the animal family", "edited object": "animal family", "output": "studio background."} """,
            """
            [INST]User: "Baked Salmon With Bell Peppers"[/INST] 
            Assistant: {"edit": "eliminate Bell Peppers", "edited object": "Bell Peppers", "output": "Baked Salmon"} """,
            """
            [INST]User: "An airplane is flying in the sky in rainy day."[/INST] 
            Assistant: {"edit": "discard the airplane", "edited object": "airplane", "output": "the sky in rainy day."} """,
            """
            [INST]User: "photo of mountains and trees"[/INST] 
            Assistant: {"edit": "drop trees", "edited object": "trees", "output": "photo of mountains"} """,
            """
            [INST]User: "Attic Bedroom With Large Ceilings"[/INST] 
            Assistant: {"edit": "without large ceilings", "edited object": "ceilings", "output": "Attic Bedroom Without Large Ceilings"} """,
            """
            [INST]User: "Wedding rings and yellow flower on a red background"[/INST]  
            Assistant: {"edit": "erase yellow flower", "edited object": "yellow flower", "output": "Wedding rings on a red background"} """,
            """
            [INST]User: "Tree Near the lake in the morning"[/INST] 
            Assistant: {"edit": "take out tree", "edited object": "tree", "output": "a lake in the morning"} """,
            """
            [INST]User: "robot and alien sitting on hanging bridge at daytime"[/INST]  
            Assistant: {"edit": "eliminate robot.", "edited object": "robot", "output": "alien sitting on hanging bridge at daytime"} """,
            """
            [INST]User: "Polar Bear with rubber gloves pushing shopping carts"[/INST]  
            Assistant: {"edit": "remove rubber gloves", "edited object": "rubber gloves", "output": "Polar Bear pushing shopping carts"}
            """
            ]
        optional_verbs = ["remove", "exclude", "delete", "erase", "omit", "take out", "eliminate", "without", "discard", "drop"]
        # system message 
        system_message = f"""
        <<SYS>>You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is 
            JSON with the following string fields: 'edit', 'edited object','output'. 
            Here is the latest conversation between Assistant and User.
        <</SYS>>
        """

        # introduction
        intro_message = f"""
        [INST]User: Hi, My job to take a given caption ('input') and to output the following: an instruction for {optional_verbs} an object from the given caption, the object to {optional_verbs} ('edited object'), and the caption without the object ('output'). Please help me do it.
            I will give you the 'input', and you will help. When you reply, use the following format: {{"edit": '<instruction>', 'edited object': '<object>', 'output': '<caption>'}}
        [/INST]
        Assistant: Sure, I'd be happy to help! Please provide the actual input caption you'd like me to read and I'll assist you with writing an instruction to {optional_verbs} the object from the given caption, writing the removed object and writing the caption without the object."""  
        
        # shuffling #
        random.seed(torch.randint(1 << 32, ()).item())
        shuffle(few_shot_examples)
        few_shot_examples = few_shot_examples[:int(len(few_shot_examples) * 0.6)]
        prompt = system_message + intro_message + "".join(few_shot_examples)        # add the test prompt
        prompt = prompt + f"[INST]User: {new_prompt}[/INST]"

        return prompt

    def generate(self, prompt,task='add'):
        if 'add' in task:
            messages = self.get_content_instruction(prompt)
        elif 'remove' in task:
            messages = self.del_content_instruction(prompt)
        outputs = self.pipeline(
            messages,
            # temperature=0.9,
            # top_p=0.9,
        )
        generated_text = outputs[0]["generated_text"]
        instruction = generated_text.split(""""edit":""")[-1].strip()
        instruction = instruction.split(",")[0].strip()
        return instruction


class Qwen_Instruction():
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


    def change_background(self,):
        system_prompt = """
        You are an AI image editing assistant. Given the prompt and its corresponding image (maybe only have one), you need to analyze the image foreground object and background, generate precise instructions for changing the image backgrounds.
        Your final task is to output a text description instruction to change the image background from its original background to new background.

        Example Prompt:
        "A cat sitting on a wooden table in a cozy room with warm lighting."

        Example Output:
        "make the light cold and room cluttered"

        """
        return system_prompt

    def text_editing(self,):
        system_prompt = """
        You are an AI image editing assistant. Given the prompt and its corresponding image (maybe only have one), you need to analyze the text content in the image, generate precise instructions for replace the text contents.
        Your final task is to output a text description instruction to replace the text content in the image from its original text from prompt to new text. The new text can be any other words or sentence with different meaning from original ones. 

        Example Prompt:
        "A glowing blue sphere with floating metallic rings with text 'alien'."

        Example Output:
        "replace 'alien' with 'pollux'"

        """
        return system_prompt


    def pil_to_base64(self, image_path):
        image = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


    def generate(self, img_path, prompt, task):
        if 'background' in task:
            system_prompt = self.change_background()
        elif 'text' in task:
            system_prompt = self.text_editing()
        if os.path.isfile(img_path):
            image = self.pil_to_base64(img_path)
            messages = [[
                        {"role": "system", "content": system_prompt},
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
        else:
            messages = [[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f'The prompt is {prompt}.',
                                },
                            ],
                        },
                    ]]
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
        )[0]
        return output_text




def instruction_generation(task):
    if task in ['add', 'remove']:
        return Llama_Instruction()
    if task in ['background', 'text']:
        return Qwen_Instruction()



if __name__ == "__main__":
    pipe = instruction_generation(task='text')
    prompt = "a realistic image contains steam with text 'steam'."
    image = "/mnt/pollux/wentian/image_edit/steam.png"
    instruction = pipe.generate(image, prompt, task='text')
    print (instruction)


    ## CUDA_VISIBLE_DEVICES=0 python instruction_generation.py

    """
    # Environments
        pip install torch==2.5.1 torchvision==0.20.1
        pip install transformers==4.49.0
        pip install safetensors qwen_vl_utils accelerate
        pip install liger_kernel einops

    # Example 1
        # Input

            prompt = "A glowing blue sphere with floating metallic rings"
            task = 'add'
            
            pipe = instruction_generation(task=task)
            for i in range(10):
                instruction = pipe.generate(prompt, task=task)
                print (instruction)

        # Output:
            "add a colorful rainbow."
            "make a robot stand next to the sphere"
            "put a blue planet in the background"
            "make the sphere have a yellow ring."
            "add a dog and a cat"
            "Set a firework in the back"
            "add a flying saucer"
            "place a glowing blue sphere with floating metallic rings in the middle of the
            "Make it wear a helmet"
            "make the sphere a little bit bigger"

    # Example 2

        # Input

            prompt = "A cute creature sits at the beach."
            image_path = '/mnt/pollux/wentian/image_edit/duck.jpeg'
            task = 'background'

            pipe = instruction_generation(task=task)
            instruction = pipe.generate(image_path, prompt, task=task)
            print (instruction)

        # Output:
            "change the beach to a snowy mountain landscape"

    # Example 3

        # Input
            prompt = "a realistic image contains steam with text 'steam'."
            image = "/mnt/pollux/wentian/image_edit/steam.png"
            task = 'text'

            pipe = instruction_generation(task='text')
            instruction = pipe.generate(image, prompt, task='text')
            print (instruction)

        # Output:
            "replace 'steam' with 'vapor'"
    """

