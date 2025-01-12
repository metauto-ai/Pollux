import wandb
from PIL import Image
import io


class WandBLogger:
    def __init__(self, project: str, run_name: str, entity: str = None):
        """
        Initialize the WandBLogger class.

        Args:
            project (str): The W&B project name.
            run_name (str): The name of the run.
            entity (str): The W&B entity (team or account). Optional.
        """
        self.run = wandb.init(project=project, name=run_name, entity=entity)
        self.images = []

    def add_image(self, image_bytes: bytes, caption: str):
        """
        Add an image to the list of images to be logged later.

        Args:
            image_bytes (bytes): Raw image bytes (e.g., from a buffer).
            caption (str): Caption for the image.
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        self.images.append(wandb.Image(image, caption=caption))

    def log_images(self, log_key: str = "images"):
        """
        Log all gathered images to W&B at once.

        Args:
            log_key (str): The key under which the images are logged. Default is "images".
        """
        if self.images:
            wandb.log({log_key: self.images})
            self.images = []  # Clear the list after logging

    def log_image(self, image_bytes: bytes, caption: str, log_key: str = "image"):
        """
        Log an image with a caption to W&B.

        Args:
            image_path (str): Path to the image file.
            caption (str): Caption for the image.
            log_key (str): The key under which the image is logged. Default is "image".
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        wandb.log({log_key: wandb.Image(image, caption=caption)})

    def finish(self):
        """Finish the W&B run."""
        wandb.finish()
