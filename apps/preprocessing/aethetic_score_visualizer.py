from pymongo import MongoClient
import certifi
from apps.main.utils.mongodb_data_load import MONGODB_URI
import requests
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from apps.preprocessing.wandb_img import WandBLogger
from apps.preprocessing.MongoDB_visualizer import MongoDBVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class MongoDBAetheticVisualizer(MongoDBVisualizer):
    def __init__(
        self,
        collection_name,
        media_field,
        other_fields,
        batch_size,
        run_name,
        max_workers,
        score_field,
    ):
        super().__init__(
            collection_name=collection_name,
            media_field=media_field,
            other_fields=other_fields,
            batch_size=batch_size,
            run_name=run_name,
            max_workers=max_workers,
        )
        self.score_field = score_field

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

    def collect(self, min_aesthetic_score, max_aesthetic_score):

        docs = list(
            self.collection.find(
                {
                    f"{self.score_field}": {
                        "$exists": True,
                        "$gt": float(min_aesthetic_score),
                        "$lt": float(max_aesthetic_score),
                        # "$elemMatch": {
                        #     "$gt": min_aesthetic_score,
                        #     "$lt": max_aesthetic_score,
                        # }
                    }
                }
            ).limit(self.batch_size)
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(self.run, doc): doc["_id"] for doc in docs}
            for future in as_completed(future_to_id):
                _id = future_to_id[future]
                try:
                    caption, image_bytes = future.result()
                    self.wandb_logger.add_image(image_bytes, str(caption))
                except Exception as e:
                    logging.warning(f"Erros in handling {_id}:{e}")

    def visualize(self, log_key="images"):
        self.wandb_logger.log_images(log_key=log_key)


if __name__ == "__main__":
    visualizer = MongoDBAetheticVisualizer(
        collection_name="cc12m",
        media_field="s3url",
        other_fields=["aesthetic_score"],
        batch_size=100,
        max_workers=100,
        run_name="cc12m_aethetic_score_visualization",
        score_field="aesthetic_score",
    )
    visualizer.collect(0, 3.0)
    visualizer.visualize("aethetic-score: <3.0")
    visualizer.collect(3.0, 4.0)
    visualizer.visualize("aethetic-score: 3.0-4.0")
    visualizer.collect(4.0, 5.0)
    visualizer.visualize("aethetic-score: 4.0-4.5")
    visualizer.collect(50, 5.5)
    visualizer.visualize("aethetic-score: 5.0-5.5")
    visualizer.collect(5.5, 6.0)
    visualizer.visualize("aethetic-score: 5.5-6.0")
    visualizer.collect(6.0, 6.5)
    visualizer.visualize("aethetic-score: 6.0-6.5")
    visualizer.collect(6.5, 7.0)
    visualizer.visualize("aethetic-score: 6.5-7.0")
    visualizer.collect(7.0, 100)
    visualizer.visualize("aethetic-score: >7.0")
    visualizer.finish()
