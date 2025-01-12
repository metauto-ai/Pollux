from pymongo import MongoClient
import certifi
from apps.main.utils.mongodb_data_load import MONGODB_URI
from apps.preprocessing.wandb_img import WandBLogger
import requests
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class MongoDBVisualizer:
    def __init__(
        self,
        collection_name,
        media_field,
        other_fields,
        batch_size,
        run_name,
        max_workers,
    ):
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        self.collection = db[collection_name]
        self.batch_size = batch_size
        self.media_field = media_field
        self.other_fields = other_fields
        self.wandb_logger = WandBLogger(
            project="pollux", run_name=run_name, entity="metauto"
        )
        self.max_workers = max_workers

    def random_sample(self, n):
        return list(self.collection.aggregate([{"$sample": {"size": n}}]))

    def process_image(self, image):
        width, height = image.size
        min_dim = min(width, height)

        if min_dim > 512:
            scale = 512 / min_dim
        elif min_dim < 512:
            scale = 256 / min_dim
        else:
            # If the size is already 512, return the original image
            return image

        # Calculate new dimensions while keeping the aspect ratio
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def collect(self):
        docs = self.random_sample(self.batch_size)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(self.run, doc): doc["_id"] for doc in docs}
            for future in as_completed(future_to_id):
                _id = future_to_id[future]
                try:
                    caption, image_bytes = future.result()
                    self.wandb_logger.add_image(image_bytes, str(caption))
                except Exception as e:
                    logging.warning(f"Erros in handling {_id}:{e}")

    def run(self, doc):
        media = doc[self.media_field]
        response = requests.get(media)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = self.process_image(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        other = {k: doc[k] for k in self.other_fields}
        return other, image_bytes

    def visualize(self):
        self.wandb_logger.log_images()

    def finish(self):
        self.wandb_logger.finish()


if __name__ == "__main__":
    # visualizer = MongoDBVisualizer(
    #     collection_name="big35m_new",
    #     media_field="azure_url",
    #     other_fields=["text", "aesthetic_score"],
    #     batch_size=100,
    #     max_workers=100,
    #     run_name="big35m_new_visualization",
    # )
    visualizer = MongoDBVisualizer(
        collection_name="laion12m_new",
        media_field="s3url",
        other_fields=[
            "TEXT",
            "pwatermark",
            "AESTHETIC_SCORE",
        ],
        batch_size=100,
        max_workers=100,
        run_name="laion12m_visualization",
    )
    for _ in range(1):
        visualizer.collect()
        visualizer.visualize()
    visualizer.finish()
