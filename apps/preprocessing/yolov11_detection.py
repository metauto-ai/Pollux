import os
import io
import time
import logging
import certifi
import requests

from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient
from ultralytics import YOLO
from apps.main.utils.mongodb_data_load import MONGODB_URI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

Image.MAX_IMAGE_PIXELS = None


class YOLOv11Detection:
    def __init__(
        self,
        collection_name: str,
        image_field: str,
        detection_field: str = "detections",
        model_path: str = "/jfs/checkpoints/yolov11/yolo11x.pt",
        conf_threshold: float = 0.15,
        max_workers: int = 32,
        batch_size: int = 500,
    ):

        # -------- MongoDB --------
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]  

        self.collection = db[collection_name]
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        self.image_field = image_field
        self.detection_field = detection_field
        self.batch_size = batch_size
        self.max_workers = max_workers

    def run(self):

        docs = self.read_data_from_mongoDB()
        if not docs:
            logging.info("No new documents found to process.")
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_docid = {
                executor.submit(self.batch_process, doc): doc["_id"]
                for doc in docs
            }

            for future in as_completed(future_to_docid):
                _id = future_to_docid[future]
                try:
                    detections_list = future.result()
                except Exception as e:
                    logging.warning(f"Error in handling {_id}: {e}")
                else:
                    self.update_data(_id, detections_list)

    def read_data_from_mongoDB(self):

        query = {f"{self.detection_field}": {"$exists": False}}
        cursor = self.collection.find(query).limit(self.batch_size)
        return list(cursor)

    def download_image(self, url):
 
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image

    def process_image(self, image):

        width, height = image.size
        min_dim = min(width, height)
        if min_dim > 1000:
            scale = 1000 / min_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return resized_image
        return image

    def batch_process(self, doc):
   
        img_url = doc[self.image_field]
        pil_image = self.download_image(img_url)
        pil_image = self.process_image(pil_image)
        results = self.model.predict(source=pil_image, conf=self.conf_threshold)

        detections_list = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id] 
                xyxy = box.xyxy[0].tolist()         
                conf = float(box.conf[0])          

                detections_list.append(
                    {
                        "class_name": cls_name,
                        "xyxy": [
                            round(xyxy[0], 2),
                            round(xyxy[1], 2),
                            round(xyxy[2], 2),
                            round(xyxy[3], 2),
                        ],
                        "confidence": round(conf, 3),
                    }
                )
        return detections_list

    def update_data(self, _id, detections_list):

        query = {"_id": _id}
        update = {"$set": {f"{self.detection_field}": detections_list}}
        self.collection.update_one(query, update)


if __name__ == "__main__":

    batch_size = 200
    max_samples_per_min = 500
    detector = YOLOv11Detection(
        collection_name="unsplash_images",  
        image_field="s3url",               
        detection_field="yolov11_detection", 
        model_path="/jfs/checkpoints/yolov11/yolo11x.pt",
        conf_threshold=0.15,
        max_workers=batch_size,
        batch_size=batch_size,
    )

    start_time = time.time()
    processed_samples = 0
    total_samples = 0

    while True:
        detector.run() 
        elapsed_time = time.time() - start_time
        processed_samples += batch_size
        total_samples += batch_size

        if processed_samples >= max_samples_per_min:

            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                logging.info(f"Sleeping for {sleep_time} seconds to respect rate-limit")
                time.sleep(sleep_time)

            start_time = time.time()
            processed_samples = 0

        logging.info(f"Total samples processed: {total_samples}")
