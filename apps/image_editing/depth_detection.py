import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import os
from tqdm import tqdm

class Depth_Pro():
    def __init__(self,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = DepthProImageProcessorFast.from_pretrained("/mnt/pollux/checkpoints/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("/mnt/pollux/checkpoints/DepthPro-hf").to(self.device)

    def generate(self,image_path_list, save_dir):
        for path in tqdm(image_path_list):
            image = Image.open(path)
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(image.height, image.width)],
            )
            depth = post_processed_output[0]["predicted_depth"]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth * 255.
            depth = depth.detach().cpu().numpy()
            depth = Image.fromarray(depth.astype("uint8"))

            filename = os.path.basename(path)
            save_path = os.path.join(save_dir,filename)
            depth.save(save_path)


if __name__ == '__main__':
    model = Depth_Pro()
    
    """ 
    Example:
        Input:

            image_ = ['/mnt/pollux/wentian/image_edit/images/tiger.jpg']
            model = Depth_Pro()
            model.generate(image_,'/mnt/pollux/wentian/image_edit/')

        Output:
            '/mnt/pollux/wentian/image_edit/tiger.jpg'

    """