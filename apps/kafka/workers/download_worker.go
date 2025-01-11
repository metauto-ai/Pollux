package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"sync"
	"time"

	"github.com/IBM/sarama"
)

const (
	TIMEOUT = 80 * time.Second
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Load config
	config := load_yaml_config("configs/example_config.yaml")

	// Configure Sarama
	saramaConfig := sarama.NewConfig()
	saramaConfig.Producer.Return.Successes = false
	saramaConfig.Producer.Return.Errors = false
	saramaConfig.Producer.RequiredAcks = sarama.NoResponse
	saramaConfig.Producer.Compression = sarama.CompressionGZIP
	saramaConfig.Producer.MaxMessageBytes = 1000000000 // 1GB
	saramaConfig.Version = sarama.V0_11_0_2

	// Create sync producer
	producer, err := sarama.NewAsyncProducer([]string{"localhost:9092"}, saramaConfig)
	if err != nil {
		log.Fatal("Failed to create producer:", err)
	}
	defer producer.Close()

	// Create consumer
	consumer, err := sarama.NewConsumer([]string{"localhost:9092"}, saramaConfig)
	if err != nil {
		log.Fatal("Failed to create consumer:", err)
	}
	defer consumer.Close()

	// Create worker pool
	// numConsumerPartitions := config.KafkaTopics["database"].Partitions
	// numConsumerPartitions := 1
	consumerPartitions, err := consumer.Partitions(config.KafkaTopics["database"].Name)
	if err != nil {
		log.Fatal("Failed to get partitions:", err)
	}
	producerPartitions := config.KafkaTopics["downloadedImages"].Partitions
	var wg sync.WaitGroup

	producerConfig := ProducerConfig{
		Name:       "downloadedImages",
		Partitions: producerPartitions,
		Producer:   producer,
	}

	// Start a worker for each partition
	for _, partition := range consumerPartitions {
		wg.Add(1)
		go func(partition int32) {
			defer wg.Done()
			partitionConsumer, err := consumer.ConsumePartition(
				config.KafkaTopics["database"].Name,
				partition,
				sarama.OffsetNewest,
			)
			if err != nil {
				log.Printf("Failed to create partition consumer for partition %d: %v", partition, err)
				return
			}
			defer partitionConsumer.Close()
			worker(ctx, partitionConsumer, partition, producerConfig)
		}(partition)
	}

	// Wait for all workers to complete
	wg.Wait()
}

func worker(ctx context.Context, consumer sarama.PartitionConsumer, consumerPartition int32,
	producerConfig ProducerConfig) {
	log.Printf("Worker started for partition %d", consumerPartition)
	msgChan := make(chan *sarama.ConsumerMessage, 1024)

	// Read messages from Kafka in parallel and push to channel
	go func() {
		for {
			select {
			case msg := <-consumer.Messages():
				msgChan <- msg
			case <-ctx.Done():
				close(msgChan)
				return
			}
		}
	}()

	producerPartitions := producerConfig.Partitions
	iters := 0
	for {
		select {
		case msg := <-msgChan:
			// Process message here
			var data MessageContent
			if err := json.Unmarshal(msg.Value, &data); err != nil {
				log.Printf("Error unmarshaling JSON: %v", err)
				continue
			}

			newProducerConfig := ProducerConfig{
				Name:       producerConfig.Name,
				Partitions: iters % producerPartitions,
				Producer:   producerConfig.Producer,
			}
			err := downloadAndSend(ctx, consumerPartition, newProducerConfig, data)
			if err != nil {
				log.Printf("Error processing image: %v", err)
				continue
			}
			iters++
		case <-ctx.Done():
			return
		}
	}
}

func downloadAndSend(ctx context.Context, consumerPartition int32, producerConfig ProducerConfig, data MessageContent) error {
	start := time.Now()
	results := downloadImages(ctx, producerConfig, data)

	successful := 0
	failed := 0

	totalSize := 0
	// Process results
	for i := 0; i < len(data.ImageURLs); i++ {
		result := <-results
		if result.Success {
			successful++
			totalSize += result.Size
		} else {
			failed++
			// log.Printf("Failed to download %s: %v", result.URL, result.Error)
		}
	}

	close(results)
	duration := time.Since(start)
	log.Printf("Rank %d: Downloaded %d images (%d successful, %d failed) in %.2f seconds, downloaded %s\n",
		consumerPartition, len(data.ImageURLs), successful, failed, duration.Seconds(), humanReadableSize(totalSize))

	return nil
}

func downloadImages(ctx context.Context, producerConfig ProducerConfig, data MessageContent) chan DownloadResult {
	numImages := len(data.ImageURLs)
	results := make(chan DownloadResult, numImages)
	jobs := make(chan Job, numImages)

	// Start workers
	for i := 0; i < numImages; i++ {
		go func() {
			for job := range jobs {
				select {
				case results <- downloadImage(ctx, producerConfig, job.DocumentID, job.URL):
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	// Send jobs
	go func() {
		for i := range data.ImageURLs {
			job := Job{
				URL:        data.ImageURLs[i],
				DocumentID: data.DocumentIDs[i],
			}

			select {
			case jobs <- job:
			case <-ctx.Done():
				break
			}
		}
		close(jobs)
	}()

	return results
}

func downloadImage(ctx context.Context, producerConfig ProducerConfig, documentID string, url string) DownloadResult {
	client := &http.Client{Timeout: TIMEOUT}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return DownloadResult{URL: url, Success: false, Error: err}
	}

	resp, err := client.Do(req)
	if err != nil {
		return DownloadResult{URL: url, Success: false, Error: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return DownloadResult{URL: url, Success: false, Error: fmt.Errorf("HTTP %d", resp.StatusCode)}
	}

	// Use a streaming encoder to directly encode base64 while reading the image
	var base64Buffer bytes.Buffer
	base64Writer := base64.NewEncoder(base64.StdEncoding, &base64Buffer)
	if _, err := io.Copy(base64Writer, resp.Body); err != nil {
		return DownloadResult{URL: url, Success: false, Error: err}
	}
	base64Writer.Close() // Ensure the base64 encoder flushes any remaining data

	payload := MessagePayload{
		DocumentIDs:   []string{documentID},
		ImageContents: []string{base64Buffer.String()}, // Avoid additional memory allocations
	}

	jsonBuffer := &bytes.Buffer{}
	encoder := json.NewEncoder(jsonBuffer)
	if err := encoder.Encode(payload); err != nil {
		return DownloadResult{URL: url, Success: false, Error: err}
	}

	producerConfig.Producer.Input() <- &sarama.ProducerMessage{
		Topic:     "downloadedImages",
		Key:       sarama.StringEncoder(fmt.Sprintf("partition-%d", producerConfig.Partitions)),
		Value:     sarama.ByteEncoder(jsonBuffer.Bytes()),
		Partition: int32(producerConfig.Partitions),
	}

	return DownloadResult{URL: url, Success: true, Error: nil, Size: jsonBuffer.Len()}
}
