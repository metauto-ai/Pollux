"""
https://github.com/discus0434/aesthetic-predictor-v2-5



            
"""

import io
import os
import requests
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Final
from tqdm import tqdm
from PIL import Image
import json
from multiprocessing import Pool, cpu_count


load_dotenv()

MONGODB_URI: Final[str] = os.environ["MONGODB_URI"]
MONGODB_USER: Final[str] = os.environ["MONGODB_USER"]
MONGODB_PASSWORD: Final[str] = os.environ["MONGODB_PASSWORD"]
encoded_user = quote_plus(MONGODB_USER)
encoded_password = quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{MONGODB_URI}"

"""
Download json

mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
--db=world_model \
--collection=midjourney_discord-1 \
--out=/jfs/jinjie/code/downloads/temp_data/midjourney_discord-1.json --jsonArray

upload json

mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
--db=world_model \
--collection= midjourney_discord-1-splited \
--file=/mnt/pollux/mongo_db_cache/midjourney_discord-1.json --jsonArray

    def download_image(self, img_path):
        if img_path.startswith("s3://") or img_path.startswith("http"):
            s3url = img_path
            response = requests.get(s3url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            with open(img_path, "rb") as f:
                image = Image.open(f).convert("RGB")
        return image
        
"""

# def score_and_update_documents(collection, image_url_field):
#     # Connect to MongoDB
#     client = MongoClient(MONGODB_URI)
#     db = client["world_model"]

#     # Process in batches
#     batch_size = 100
#     while True:
#         # Find a batch of documents without aesthetic score
#         docs = collection.find(
#             {"aesthetic_score": {"$exists": False}}, batch_size=batch_size
#         ).limit(batch_size)

#         # Convert cursor to list to avoid timeout issues
#         docs = list(docs)

#         # Break if no more documents to process
#         if not docs:
#             break

#         for doc in tqdm(docs):
#             try:
#                 imageUrl = doc[image_url_field]

#                 # download the image to a tmp directory in my linux machine

#                 # Download image and load directly into PIL
#                 response = requests.get(imageUrl)
#                 image = Image.open(io.BytesIO(response.content)).convert("RGB")

#                 """
#                 # Update document with score
#                 collection.update_one(
#                     {"_id": doc["_id"]},
#                     {"$set": {"aesthetic_score": float(score)}}
#                 )
#                 """

#                 print(f"Image URL {doc[image_url_field]}")

#             except Exception as e:
#                 print(
#                     f"Error processing {doc.get(image_url_field, 'unknown')}: {str(e)}"
#                 )
#                 continue

#     client.close()

# unsplash images
# collection = db.unsplash_images
# IMAGE_URL_FIELD = "imageUrl"

# # CC12M images
# collection = db.cc12m
# IMAGE_URL_FIELD = "s3url"

# score_and_update_documents(collection, IMAGE_URL_FIELD)


"""
List of 1227317 items
    _id: dict
        $oid: str
    filename: str
    url: str
    caption: str
    source: str
    is_collage: bool
    message_id: str
    width: int
    height: int
    author: dict
        id: str
        username: str
    timestamp: str
    parsed_prompt: dict
        main_prompt: str
        version: float
        aspect_ratio: str
        style: NoneType
        seed: int
        chaos: NoneType
        creative_weight: NoneType
        reference_images: list
            List of 0 items
"""


def download_and_split_image(args):
    record, jfs_export_path=args
    assert record.get("is_collage", False) == True
    img_path = record["url"]
    if img_path.startswith("s3://") or img_path.startswith("http"):
        s3url = img_path
        response = requests.get(s3url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

    # split the image into 4 parts

    width, height = image.size
    # assert width==record['width'] and height==record['height']
    if width!=record['width'] or height!=record['height']:
        print(
            f"Warning : Image {record['_id']['$oid']} size mismatch: actual {width}x{height} != record {record['width']}x{record['height']}"
        )
    mid_width, mid_height = width // 2, height // 2

    # Define the box coordinates for each quadrant
    boxes = [
        (0, 0, mid_width, mid_height),  # Top-left
        (mid_width, 0, width, mid_height),  # Top-right
        (0, mid_height, mid_width, height),  # Bottom-left
        (mid_width, mid_height, width, height),  # Bottom-right
    ]

    # Crop the image into 4 parts
    split_images = [image.crop(box) for box in boxes]
    new_records = [record.copy() for _ in range(4)]

    for i, new_record in enumerate(new_records):
        # avoid some weird . in the filename
        name, ext = os.path.splitext(record['filename'])
        temp_jfs_path=os.path.join(jfs_export_path, f"{name}_split_{i}{ext}")
        # save the image to file path
        split_images[i].save(temp_jfs_path)
        new_record['_id'] = {'$oid':record['_id']['$oid'] + f"_{i}"}
        new_record["url"] = temp_jfs_path
        new_record["width"] = split_images[i].width
        new_record["height"] = split_images[i].height
        new_record["is_collage"] = False

    return new_records


if __name__ == "__main__":
    # Connect to MongoDB
    # client = MongoClient(MONGODB_URI)
    # db = client["world_model"]

    # midjourney images
    # collection = db["midjourney_discord-1"]
    IMAGE_URL_FIELD = "url"
    json_path="/mnt/pollux/mongo_db_cache/midjourney_discord-1.json"
    new_json_path="/mnt/pollux/mongo_db_cache/midjourney_discord-1-splited.json"
    new_json_data=[]
    # new_collection = db["midjourney_discord-1-splited"]
    jfs_export_path="/jfs/jinjie/code/downloads/mid_journey_splited"

    with open(json_path, 'r') as file:
        data = json.load(file)

    records_to_process=[]
    for record in data:
        if record.get("is_collage", False):
            records_to_process.append(record)

    # * demo test
    # records_to_process = records_to_process[:100]

    # Use multiprocessing to process records in parallel
    with Pool(processes=32) as pool:
        results = list(
            tqdm(
                pool.imap(
                    download_and_split_image,
                    [(record, jfs_export_path) for record in records_to_process],
                ),
                total=len(records_to_process),
            )
        )

    for new_records in results:
        new_json_data.extend(new_records)

    with open(new_json_path, 'w') as file:
        json.dump(new_json_data, file, indent=4)

    print("All finished")

"""
example split json output will look like:

[
    {
        "_id": {
            "$oid": "678586c60280fa5f0da6f999_0"
        },
        "filename": "elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637.png",
        "url": "/jfs/jinjie/code/downloads/mid_journey_splited/elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637_split_0.png",
        "caption": "**Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style. --ar 3:4 --s 750 --v 5.1** - <@1114245101292617738> (relaxed)",
        "source": "discord",
        "is_collage": false,
        "message_id": "1200915561928998952",
        "width": 928,
        "height": 1232,
        "author": {
            "id": "936929561302675456",
            "username": "Midjourney Bot"
        },
        "timestamp": "2024-01-27T21:29:41.896000+00:00",
        "parsed_prompt": {
            "main_prompt": "Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style.",
            "version": 5.1,
            "aspect_ratio": "3:4",
            "style": null,
            "seed": 750,
            "chaos": null,
            "creative_weight": null,
            "reference_images": []
        }
    },
    {
        "_id": {
            "$oid": "678586c60280fa5f0da6f999_1"
        },
        "filename": "elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637.png",
        "url": "/jfs/jinjie/code/downloads/mid_journey_splited/elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637_split_1.png",
        "caption": "**Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style. --ar 3:4 --s 750 --v 5.1** - <@1114245101292617738> (relaxed)",
        "source": "discord",
        "is_collage": false,
        "message_id": "1200915561928998952",
        "width": 928,
        "height": 1232,
        "author": {
            "id": "936929561302675456",
            "username": "Midjourney Bot"
        },
        "timestamp": "2024-01-27T21:29:41.896000+00:00",
        "parsed_prompt": {
            "main_prompt": "Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style.",
            "version": 5.1,
            "aspect_ratio": "3:4",
            "style": null,
            "seed": 750,
            "chaos": null,
            "creative_weight": null,
            "reference_images": []
        }
    },
    {
        "_id": {
            "$oid": "678586c60280fa5f0da6f999_2"
        },
        "filename": "elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637.png",
        "url": "/jfs/jinjie/code/downloads/mid_journey_splited/elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637_split_2.png",
        "caption": "**Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style. --ar 3:4 --s 750 --v 5.1** - <@1114245101292617738> (relaxed)",
        "source": "discord",
        "is_collage": false,
        "message_id": "1200915561928998952",
        "width": 928,
        "height": 1232,
        "author": {
            "id": "936929561302675456",
            "username": "Midjourney Bot"
        },
        "timestamp": "2024-01-27T21:29:41.896000+00:00",
        "parsed_prompt": {
            "main_prompt": "Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style.",
            "version": 5.1,
            "aspect_ratio": "3:4",
            "style": null,
            "seed": 750,
            "chaos": null,
            "creative_weight": null,
            "reference_images": []
        }
    },
    {
        "_id": {
            "$oid": "678586c60280fa5f0da6f999_3"
        },
        "filename": "elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637.png",
        "url": "/jfs/jinjie/code/downloads/mid_journey_splited/elendilka_Flight_attendants_on_the_plane_using_statistics_metri_a65d574f-0d34-4607-b5ea-247752df2637_split_3.png",
        "caption": "**Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style. --ar 3:4 --s 750 --v 5.1** - <@1114245101292617738> (relaxed)",
        "source": "discord",
        "is_collage": false,
        "message_id": "1200915561928998952",
        "width": 928,
        "height": 1232,
        "author": {
            "id": "936929561302675456",
            "username": "Midjourney Bot"
        },
        "timestamp": "2024-01-27T21:29:41.896000+00:00",
        "parsed_prompt": {
            "main_prompt": "Flight attendants on the plane using statistics, metrics, results, and an icon. white background is a white background. Bright, clear details. Manga style.",
            "version": 5.1,
            "aspect_ratio": "3:4",
            "style": null,
            "seed": 750,
            "chaos": null,
            "creative_weight": null,
            "reference_images": []
        }
    },
    {
        "_id": {
            "$oid": "678586c70280fa5f0da6f99c_0"
        },
        "filename": "elendilka_Giraffe_in_safari_using_statistics_metrics_results_ic_7086d623-252c-4558-a31d-80eb49dbf914.png",
        "url": "/jfs/jinjie/code/downloads/mid_journey_splited/elendilka_Giraffe_in_safari_using_statistics_metrics_results_ic_7086d623-252c-4558-a31d-80eb49dbf914_split_0.png",
        "caption": "**Giraffe in safari using statistics, metrics, results, icon, white background White background. Bright, clear details. Manga style. --ar 3:4 --s 750 --v 5.1** - <@1114245101292617738> (relaxed)",
        "source": "discord",
        "is_collage": false,
        "message_id": "1200912842740748298",
        "width": 928,
        "height": 1232,
        "author": {
            "id": "936929561302675456",
            "username": "Midjourney Bot"
        },
        "timestamp": "2024-01-27T21:18:53.591000+00:00",
        "parsed_prompt": {
            "main_prompt": "Giraffe in safari using statistics, metrics, results, icon, white background White background. Bright, clear details. Manga style.",
            "version": 5.1,
            "aspect_ratio": "3:4",
            "style": null,
            "seed": 750,
            "chaos": null,
            "creative_weight": null,
            "reference_images": []
        }
    },

"""
