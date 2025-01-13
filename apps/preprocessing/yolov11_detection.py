import cv2
from PIL import Image
from ultralytics import YOLO

# TODO: need to organize the label and spatial info; and make this multiprocessing
model = YOLO("/jfs/checkpoints/yolov11/yolo11x.pt")
image = Image.open("GhAc5VUWQAAe3Bp.jpeg")
results = model.predict(source=image, save=True, save_txt=True) 
# results[0].show()

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])