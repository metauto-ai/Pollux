# IndusFlow 

This is an abstract pipeline designed to efficiently distribute workloads for querying data from a database. It utilizes data loaders to perform inference on the retrieved data and subsequently updates the database through Kafka and Go downloads.

## Kafka Setup Reference [Link](Kafka_Setup.md)

## Environment Setup

```bash
pip install -r requirements.txt
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

## Usage

### Example Config to Setup Kafka Topics and Stages

```yaml
kafka_topics: # These are the topics that will be used in the pipeline
  database: # This is the topic that will be used to produce data to the database 
    name: database # This is the name of the topic
    partitions: 16 # This is the number of partitions in the topic
  downloadedImages: 
    name: downloadedImages
    partitions: 32
  aestheticScoredImages:
    name: aestheticScoredImages
    partitions: 16

stages: # These are the stages that will be executed in the pipeline
  read_database: # This stage reads data from the database and produces it to the database topic
    batch_size: 32 # This is the batch size that will be sent to the next stage
    dataset: diffusion # This is the dataset that will be used to read data from the database
    producer_list:
      - database # This is the topic that will be used to produce data to the database

  download_images: # This stage downloads images from the database and produces it to the downloadedImages topic
    batch_size: 64    # per database partition
    consumer: database
    producer_list:
      - downloadedImages

  aesthetic_scoring: # This stage performs aesthetic scoring on the downloaded images and produces it to the aestheticScoredImages topic
    batch_size: 128
    consumer: downloadedImages
    producer_list:
      - aestheticScoredImages

  update_database: # This stage updates the database with the aesthetic scores
    dataset: diffusion
    consumer: aestheticScoredImages
```

Note: 
- The config is designed to be modular and can be extended to include more stages.
- For optimal performance, read_database batch size * database partitions should be less than 16,000 (This is the number of parallel connections that can be made to download images). 
    - Ideally a system has around 64000 ephemeral ports, all these ports can be used if image size is small(~25kb).
    - But this config also depends on the image size being downloaded.

### How to Run Each Stage

Note: Each stage has a worker script that can be run independently, in different screen/tmux sessions.

```bash
python -m workers.update_db_worker
python -m workers.aesthetic_scoring_worker
# python -m workers.download_worker
go run workers/download_worker.go workers/common.go
python -m workers.read_db_worker
```
