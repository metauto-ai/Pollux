# How to setup Kafka for the pipeline

1. [Download kafka binaries](#Step-1a-Download-kafka-binaries)
2. [Add the following scripts to system path](#Step-1b-Add-the-following-scripts-to-system-path)
3. [Setup Server, Consumer and Producer scripts](#Step-2-Setup-Server-Consumer-and-Producer-scripts)
4. [Start the Kafka server and Zookeeper](#Step-3-Start-the-Kafka-server-and-Zookeeper)
5. [Common Commands for Kafka](#Step-4-Common-Commands-for-Kafka)

## Step 1a: Download kafka binaries

```bash
mkdir kafka_binaries
wget https://downloads.apache.org/kafka/3.9.0/kafka_2.13-3.9.0.tgz
tar -xzf kafka_2.13-3.9.0.tgz
mv kafka_2.13-3.9.0 ./kafka_binaries
```

## Step 1b: Add the following scripts to system path

### Zookeeper service script
```bash
sudo vi /etc/systemd/system/zookeeper.service
```

Note: Replace the path with the path to your kafka binaries
```conf
[Unit]
Description=ZooKeeper Service
Requires=network.target
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
ExecStart= path/to/your/kafka_binaries/bin/zookeeper-server-start.sh  path/to/your/kafka_binaries/config/zookeeper.properties
ExecStop= path/to/your/kafka_binaries/bin/zookeeper-server-stop.sh
Restart=on-abnormal

[Install]
WantedBy=multi-user.target
```

### Kafka service script
```bash
sudo vi /etc/systemd/system/kafka.service
```
Note: Replace the path with the path to your kafka binaries
```conf
[Unit]
Description=Apache Kafka Server
Requires=zookeeper.service
After=zookeeper.service
StartLimitInterval=200
StartLimitBurst=5

[Service]
Type=simple
User=ubuntu
Group=ubuntu
ExecStart= path/to/your/kafka_binaries/bin/kafka-server-start.sh path/to/your/kafka_binaries/config/server.properties
ExecStop= path/to/your/kafka_binaries/bin/kafka-server-stop.sh
Restart=on-abnormal

[Install]
WantedBy=multi-user.target
```

## Step 2: Setup Consumer and Producer scripts

### Server script

```bash
sudo vi /path/to/your/kafka_binaries/config/server.properties
```

Append the following settings to the server.properties file
```conf
# Broker settings for the Message Size - Current size is 512MB, Modify as needed
message.max.bytes=536870912
replica.fetch.max.bytes=576716800

# Topic level setting (optional)
max.message.bytes=536870912

# Delete topic settings
delete.topic.enable=true
```
### Note : Edit or add log retention settings "log.retention.ms=300000". This is to ensure that the logs are deleted after 5 minutes, to avoid disk space issues.


### Consumer script

```bash
sudo vi /path/to/your/kafka_binaries/config/consumer.properties
```

Append the following settings to the consumer.properties file
```conf
# Message size settings
fetch.max.bytes=536870912
max.partition.fetch.bytes=536870912
```

### Producer script

```bash
sudo vi /path/to/your/kafka_binaries/config/producer.properties
```

Append the following settings to the producer.properties file
```conf
# the maximum size of a request in bytes
max.request.size=536870912
```

## Step 3: Start the Kafka server
```bash
sudo systemctl daemon-reload
sudo systemctl enable zookeeper
sudo systemctl enable kafka
sudo systemctl start zookeeper
```

Note: 
- "enable" Ensures that the service is enabled and will start on boot
- Wait for 10 seconds for zookeeper to start. You can check the status of the service with: 
    - sudo systemctl status zookeeper

```bash
sudo systemctl start kafka
```

## Step 4: Common Commands for Kafka 

### [Note: The topics are dynamically created in the existing worker nodes. So below commands are optional]

### Create a topic

```bash
path/to/your/kafka_binaries/bin/kafka-topics.sh --create --topic topic_name --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
```

### List all topics
```bash
path/to/your/kafka_binaries/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

### Delete a topic
```bash
path/to/your/kafka_binaries/bin/kafka-topics.sh --delete --topic topic_name --bootstrap-server localhost:9092
```

### Alter a topic
```bash
path/to/your/kafka_binaries/bin/kafka-topics.sh --alter --topic topic_name --bootstrap-server localhost:9092 --partitions 2
```

