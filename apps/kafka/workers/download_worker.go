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

	"sync/atomic"

	"github.com/IBM/sarama"
)

const (
	TIMEOUT = 80 * time.Second
)

var counter atomic.Uint32

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Load config
	config := load_yaml_config("configs/pd12m_config.yaml")

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

	consumerTopic := config.Stages["download_images"].Consumer
	producerTopic := config.Stages["download_images"].ProducerList[0]

	// Create worker pool
	numConsumerPartitions := config.KafkaTopics[consumerTopic].Partitions
	// numConsumerPartitions := 1
	producerPartitions := config.KafkaTopics[producerTopic].Partitions
	var wg sync.WaitGroup

	producerConfig := ProducerConfig{
		Name:       producerTopic,
		Partitions: producerPartitions,
		Producer:   producer,
	}

	// Track the number of messages processed
	docsChan := make(chan int, 1024)

	log.Printf("Producer Partitions: %d, Consumer Partitions: %d", producerPartitions, numConsumerPartitions)
	// Start a worker for each partition
	for partition := 0; partition < numConsumerPartitions; partition++ {
		wg.Add(1)
		go func(partition int32) {
			defer wg.Done()
			partitionConsumer, err := consumer.ConsumePartition(
				consumerTopic,
				partition,
				sarama.OffsetNewest,
			)
			if err != nil {
				log.Printf("Failed to create partition consumer for partition %d: %v", partition, err)
				return
			}
			defer partitionConsumer.Close()
			worker(ctx, partitionConsumer, partition, producerConfig, docsChan)
		}(int32(partition))
	}

	go processDocument(ctx, docsChan)

	// Wait for all workers to complete
	wg.Wait()
}

func worker(ctx context.Context, consumer sarama.PartitionConsumer, consumerPartition int32,
	producerConfig ProducerConfig, docsChan chan int) {
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

	for {
		select {
		case msg := <-msgChan:
			// Process message here
			var data MessageContent
			if err := json.Unmarshal(msg.Value, &data); err != nil {
				log.Printf("Error unmarshaling JSON: %v", err)
				continue
			}

			downloadAndSend(ctx, consumerPartition, producerConfig, data, docsChan)
		case <-ctx.Done():
			return
		}
	}
}

func processDocument(ctx context.Context, docsChan chan int) {

	totalDocuments := 0
	currProcessedDocuments := 0
	totalSize := 0
	// Create a ticker to print processed documents every minute
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case size := <-docsChan:
			totalDocuments++
			currProcessedDocuments++
			totalSize += size
		case <-ticker.C:
			// Print and reset the counter for the current interval
			fmt.Printf("\n---------------------------------------------------------------------------------------------\n")
			fmt.Printf("Processed %d documents/second of size %s in the last 10 seconds. Total processed: %d\n",
				currProcessedDocuments/10, humanReadableSize(totalSize), totalDocuments)
			fmt.Printf("---------------------------------------------------------------------------------------------")
			totalSize = 0
			currProcessedDocuments = 0
		case <-ctx.Done():
			fmt.Println("Stopping document processing...")
			return
		}
	}
}

func downloadAndSend(ctx context.Context, consumerPartition int32, producerConfig ProducerConfig, data MessageContent, docsChan chan int) {
	// start := time.Now()
	results := downloadImages(ctx, producerConfig, data, docsChan)

	// successful := 0
	// failed := 0

	// totalSize := 0
	// Process results
	docsToWait := int(float64(len(data.ImageURLs)) * 0.1)
	for i := 0; i < docsToWait; i++ {
		<-results
		// result := <-results
		// if result.Success {
		// 	successful++
		// 	totalSize += result.Size
		// } else {
		// 	failed++
		// 	// log.Printf("Failed to download %s: %v", result.URL, result.Error)
		// }
	}

	// close(results)
	// duration := time.Since(start)
	// log.Printf("Rank %d: Downloaded %d images (%d successful, %d failed) in %.2f seconds, downloaded %s\n",
	// 	consumerPartition, len(data.ImageURLs), successful, failed, duration.Seconds(), humanReadableSize(totalSize))
}

func downloadImages(ctx context.Context, producerConfig ProducerConfig, data MessageContent, docsChan chan int) chan DownloadResult {
	numImages := len(data.ImageURLs)
	results := make(chan DownloadResult, numImages)
	jobs := make(chan Job, numImages)

	// Start workers
	for i := 0; i < numImages; i++ {
		go func() {
			for job := range jobs {
				select {
				case results <- downloadImage(ctx, producerConfig, job.DocumentID, job.URL, docsChan):
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

func downloadImage(ctx context.Context, producerConfig ProducerConfig, documentID string, url string, docsChan chan int) DownloadResult {
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

	partitionNumber := int(counter.Add(1)) % producerConfig.Partitions

	producerConfig.Producer.Input() <- &sarama.ProducerMessage{
		Topic:     producerConfig.Name,
		Key:       sarama.StringEncoder(fmt.Sprintf("partition-%d", partitionNumber)),
		Value:     sarama.ByteEncoder(jsonBuffer.Bytes()),
		Partition: int32(partitionNumber),
	}

	docsChan <- jsonBuffer.Len()
	return DownloadResult{URL: url, Success: true, Error: nil, Size: jsonBuffer.Len()}
}
