import random 
import torch
import re
from PIL import Image
import os
from ultralytics import YOLO

class Yolo_Detection():
    def __init__(self, task):
        
        checkpoint = {
        'detection': '/mnt/pollux/checkpoints/YOLO11/yolo11x.pt', 
        'segmentation': '/mnt/pollux/checkpoints/YOLO11/yolo11x-seg.pt',
        'pose': '/mnt/pollux/checkpoints/YOLO11/yolo11x-pose.pt',
        }
        self.task = task
        self.model = YOLO(checkpoint[task])


    def pred(self, image_list):
        results = self.model(image_list)

        # Visualize the results
        for i, (r,img) in enumerate(zip(results,image_list)):
            # Plot results image
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            # Save results to disk
            save_path = f"{img[:-4]}_{self.task}.jpg"
            r.save(filename=save_path)


if __name__ == "__main__":


    ## CUDA_VISIBLE_DEVICES=0 python detection.py

    """
    # Environments
        pip install ultralytics

    # Examples
        # Input

            pipe = Yolo_Detection(task='detection')
            pipe.pred(['/mnt/pollux/wentian/image_edit/images/desk.jpg'])

            pipe = Yolo_Detection(task='segmentation')
            pipe.pred(['/mnt/pollux/wentian/image_edit/images/desk.jpg'])

            pipe = Yolo_Detection(task='pose')
            pipe.pred(['/mnt/pollux/wentian/image_edit/images/pose.jpg'])

        # Output:
            '/mnt/pollux/wentian/image_edit/images/desk_detection.jpg'
            '/mnt/pollux/wentian/image_edit/images/desk_segmentation.jpg'
            '/mnt/pollux/wentian/image_edit/images/pose_pose.jpg'

    """

