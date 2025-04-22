import random 
from random import choice, shuffle
import torch
import transformers

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

    def generate(self,prompt):
        messages = self.get_content_instruction(prompt)
        outputs = self.pipeline(
            messages,
            # temperature=0.9,
            # top_p=0.9,
        )
        generated_text = outputs[0]["generated_text"]
        instruction = generated_text.split(""""edit":""")[-1].strip()
        instruction = instruction.split(",")[0].strip()
        return instruction






if __name__ == "__main__":
    pipe = Llama_Instruction()
    prompt = "A glowing blue sphere with floating metallic rings"
    for i in range(10):
        instruction = pipe.generate(prompt)
        print (instruction)

    ## CUDA_VISIBLE_DEVICES=7 python instruction_generation.py

    """
    # Environments
        pip install torch
        pip install transformers

    # Example

    # Input

        prompt = "A glowing blue sphere with floating metallic rings"

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
    """