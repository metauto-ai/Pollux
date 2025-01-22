package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"image"
	_ "image/gif"  // Register GIF format
	_ "image/jpeg" // Register JPEG format
	"image/png"

	"github.com/IBM/sarama"
	"github.com/nfnt/resize"
	"gopkg.in/yaml.v2"
)

// HTTP client pool configuration
const (
	maxConcurrentDownloads = 50000 // Maximum number of concurrent downloads (ephemeral port limit)
	downloadTimeout        = 30 * time.Second
	benchmarkDuration      = 10 * time.Second
	metricsInterval        = 5 * time.Second
	maxIdleConnsPerHost    = 2000
	maxIdleConns           = 10000
	idleConnTimeout        = 90 * time.Second
	downloadWorkers        = 2000   // Number of download worker goroutines
	downloadQueueSize      = 100000 // Size of download queue channel
)

type Config struct {
	KafkaTopics map[string]TopicConfig `yaml:"kafka_topics"`
	Stages      struct {
		DownloadImages struct {
			Consumer string `yaml:"consumer"`
			Producer string `yaml:"producer"`
		} `yaml:"download_images"`
	} `yaml:"stages"`
}

type TopicConfig struct {
	Partitions int `yaml:"partitions"`
}

type Message struct {
	DocID    string `json:"doc_id"`
	ImageURL string `json:"image_url"`
	Caption  string `json:"caption"`
}

type ProcessedMessage struct {
	DocID       string `json:"doc_id"`
	ImageBase64 string `json:"image_base64"`
	Caption     string `json:"caption"`
}

type Metrics struct {
	mu                sync.RWMutex
	startTime         time.Time
	totalImages       atomic.Uint64
	totalBytes        atomic.Uint64
	currentBytes      atomic.Uint64
	currentImages     atomic.Uint64
	failedDownloads   atomic.Uint64
	lastUpdateTime    time.Time
	intervalDuration  time.Duration
	historicalSpeeds  []float64 // in Mbps
	historicalImgRate []float64 // images per second
}

func NewMetrics() *Metrics {
	return &Metrics{
		startTime:        time.Now(),
		lastUpdateTime:   time.Now(),
		intervalDuration: metricsInterval,
	}
}

func (m *Metrics) Record(bytes int) {
	m.totalBytes.Add(uint64(bytes))
	m.currentBytes.Add(uint64(bytes))
	m.totalImages.Add(1)
	m.currentImages.Add(1)
}

func (m *Metrics) RecordFailure() {
	m.failedDownloads.Add(1)
}

func (m *Metrics) UpdateAndGetStats() (float64, float64, float64, float64, uint64, float64, uint64, time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	duration := now.Sub(m.lastUpdateTime).Seconds()

	// Get current values
	currentBytes := m.currentBytes.Swap(0)
	currentImages := m.currentImages.Swap(0)
	totalImages := m.totalImages.Load()
	totalBytes := m.totalBytes.Load()
	failedDownloads := m.failedDownloads.Load()

	// Calculate current speeds
	currentMbps := float64(currentBytes*8) / duration / 1000000 // Convert to Mbps
	currentImagesPerSec := float64(currentImages) / duration

	// Update historical data
	m.historicalSpeeds = append(m.historicalSpeeds, currentMbps)
	m.historicalImgRate = append(m.historicalImgRate, currentImagesPerSec)

	// Calculate averages
	var avgSpeed, avgImgRate float64
	if len(m.historicalSpeeds) > 0 {
		for _, speed := range m.historicalSpeeds {
			avgSpeed += speed
		}
		for _, rate := range m.historicalImgRate {
			avgImgRate += rate
		}
		avgSpeed /= float64(len(m.historicalSpeeds))
		avgImgRate /= float64(len(m.historicalImgRate))
	}

	// Get total data in GB (convert bytes to GB)
	totalGB := float64(totalBytes) / (1024 * 1024 * 1024)

	m.lastUpdateTime = now

	return currentMbps, currentImagesPerSec, avgSpeed, avgImgRate, totalImages, totalGB, failedDownloads, now.Sub(m.startTime)
}

// Global variables
var (
	configFile     = flag.String("config", "config.yaml", "Path to config file")
	doBenchmark    = flag.Bool("benchmark", false, "Run consumer benchmark before processing")
	brokers        = flag.String("brokers", "localhost:9092", "Kafka broker list")
	downloadSem    = make(chan struct{}, maxConcurrentDownloads)
	partitionCount atomic.Uint32 // For round-robin partition selection
	metrics        = NewMetrics()
)

// Global HTTP client with optimized transport
var httpClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		MaxIdleConns:        maxIdleConns,
		IdleConnTimeout:     idleConnTimeout,
		DisableCompression:  false, // Enable compression for smaller downloads
		MaxConnsPerHost:     maxIdleConnsPerHost,
		DisableKeepAlives:   false, // Enable keep-alives for connection reuse
		Proxy:               http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	},
	Timeout: downloadTimeout,
}

// Buffer pool for reusing buffers
var bufferPool = sync.Pool{
	New: func() interface{} {
		return bytes.NewBuffer(make([]byte, 0, 1024*1024)) // 1MB initial capacity
	},
}

type downloadTask struct {
	msg     Message
	session sarama.ConsumerGroupSession
	message *sarama.ConsumerMessage
}

type ConsumerGroupHandler struct {
	producer      sarama.SyncProducer
	producerTopic string
	partitions    int
	benchmark     bool
	startTime     time.Time
	msgCount      atomic.Uint64
	downloadQueue chan downloadTask
	downloadWg    sync.WaitGroup
}

func NewConsumerGroupHandler(producer sarama.SyncProducer, producerTopic string, partitions int, benchmark bool) *ConsumerGroupHandler {
	h := &ConsumerGroupHandler{
		producer:      producer,
		producerTopic: producerTopic,
		partitions:    partitions,
		benchmark:     benchmark,
		downloadQueue: make(chan downloadTask, downloadQueueSize),
	}

	// Start download workers
	for i := 0; i < downloadWorkers; i++ {
		h.downloadWg.Add(1)
		go h.downloadWorker()
	}

	return h
}

func (h *ConsumerGroupHandler) downloadWorker() {
	defer h.downloadWg.Done()

	for task := range h.downloadQueue {
		// log.Printf("Starting download for URL: %s", task.msg.ImageURL)

		imageData, err := downloadImage(task.msg.ImageURL)
		if err != nil {
			// log.Printf("Error downloading image: %v", err)
			continue
		}

		// log.Printf("Successfully downloaded image for URL: %s (size: %d bytes)", task.msg.ImageURL, len(imageData))

		// Encode image to base64
		base64Data := base64.StdEncoding.EncodeToString(imageData)

		// Prepare processed message
		processed := ProcessedMessage{
			DocID:       task.msg.DocID,
			ImageBase64: base64Data,
			Caption:     task.msg.Caption,
		}

		// Serialize processed message
		value, err := json.Marshal(processed)
		if err != nil {
			// log.Printf("Error marshaling processed message: %v", err)
			continue
		}

		// Select partition in round-robin fashion
		partition := int32(partitionCount.Add(1) % uint32(h.partitions))

		// Produce message
		_, _, err = h.producer.SendMessage(&sarama.ProducerMessage{
			Topic:     h.producerTopic,
			Value:     sarama.ByteEncoder(value),
			Partition: partition,
		})
		if err != nil {
			log.Printf("Error producing message: %v", err)
			continue
		}

		task.session.MarkMessage(task.message, "")
	}
}

func (h *ConsumerGroupHandler) Setup(sarama.ConsumerGroupSession) error {
	h.startTime = time.Now()
	return nil
}

func (h *ConsumerGroupHandler) Cleanup(sarama.ConsumerGroupSession) error {
	close(h.downloadQueue)
	h.downloadWg.Wait()
	return nil
}

func (h *ConsumerGroupHandler) runBenchmark(session sarama.ConsumerGroupSession, message *sarama.ConsumerMessage) bool {
	// Initialize benchmark if this is the first message
	if h.startTime.IsZero() {
		h.startTime = time.Now()
		log.Printf("\nStarting benchmark...")
	}

	h.msgCount.Add(1)
	if time.Since(h.startTime) >= benchmarkDuration {
		count := h.msgCount.Load()
		throughput := float64(count) / benchmarkDuration.Seconds()
		log.Printf("\nBenchmark complete: processed %d messages in %v (%.2f msgs/sec)\n",
			count, benchmarkDuration, throughput)
		h.benchmark = false
		return false // Benchmark complete
	}

	session.MarkMessage(message, "")
	return true // Continue benchmarking
}

func (h *ConsumerGroupHandler) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	// Wait for the first message to start benchmarking
	firstMessage := true

	for message := range claim.Messages() {
		if h.benchmark {
			if firstMessage {
				firstMessage = false
			}
			if h.runBenchmark(session, message) {
				continue
			}
			// Reset for actual processing
			firstMessage = true
			continue
		}

		// Start metrics only after benchmarking is complete
		if firstMessage {
			metrics = NewMetrics() // Reset metrics after benchmark
			firstMessage = false
		}

		var msg Message
		if err := json.Unmarshal(message.Value, &msg); err != nil {
			log.Printf("Error unmarshaling message: %v", err)
			continue
		}

		// Send to download queue instead of processing directly
		select {
		case h.downloadQueue <- downloadTask{
			msg:     msg,
			session: session,
			message: message,
		}:
		default:
			// If queue is full, process in current goroutine
			log.Printf("Warning: Download queue full, processing in consumer goroutine")
			h.processDownload(msg, session, message)
		}
	}
	return nil
}

func (h *ConsumerGroupHandler) processDownload(msg Message, session sarama.ConsumerGroupSession, message *sarama.ConsumerMessage) {
	imageData, err := downloadImage(msg.ImageURL)
	if err != nil {
		log.Printf("Error downloading image: %v", err)
		return
	}

	// Encode image to base64
	base64Data := base64.StdEncoding.EncodeToString(imageData)

	// Prepare processed message
	processed := ProcessedMessage{
		DocID:       msg.DocID,
		ImageBase64: base64Data,
		Caption:     msg.Caption,
	}

	// Serialize processed message
	value, err := json.Marshal(processed)
	if err != nil {
		log.Printf("Error marshaling processed message: %v", err)
		return
	}

	// Select partition in round-robin fashion
	partition := int32(partitionCount.Add(1) % uint32(h.partitions))

	// Produce message
	_, _, err = h.producer.SendMessage(&sarama.ProducerMessage{
		Topic:     h.producerTopic,
		Value:     sarama.ByteEncoder(value),
		Partition: partition,
	})
	if err != nil {
		log.Printf("Error producing message: %v", err)
		return
	}

	session.MarkMessage(message, "")
}

func downloadImage(url string) ([]byte, error) {
	// Acquire semaphore slot
	downloadSem <- struct{}{}
	defer func() { <-downloadSem }()

	// Make request
	resp, err := httpClient.Get(url)
	if err != nil {
		metrics.RecordFailure()
		return nil, fmt.Errorf("HTTP GET error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		metrics.RecordFailure()
		return nil, fmt.Errorf("HTTP status error: %s", resp.Status)
	}

	// Validate content type
	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "image/") && contentType != "application/octet-stream" {
		metrics.RecordFailure()
		return nil, fmt.Errorf("invalid content type: %s", contentType)
	}

	// Read image into memory
	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		metrics.RecordFailure()
		return nil, fmt.Errorf("read error: %v", err)
	}

	// Record metrics for the original download size
	metrics.Record(len(imageData))

	// Decode and validate image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		metrics.RecordFailure()
		return nil, fmt.Errorf("invalid image format: %v", err)
	}

	// Check if resizing is needed
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	if width > 1024 || height > 1024 {
		// Calculate new dimensions while maintaining aspect ratio
		var newWidth, newHeight uint
		if width > height {
			newWidth = 320
			newHeight = uint(float64(height) * (320.0 / float64(width)))
		} else {
			newHeight = 320
			newWidth = uint(float64(width) * (320.0 / float64(height)))
		}

		// Resize the image
		img = resize.Resize(newWidth, newHeight, img, resize.Lanczos3)
	}

	// Encode as PNG
	buf := new(bytes.Buffer)
	if err := png.Encode(buf, img); err != nil {
		metrics.RecordFailure()
		return nil, fmt.Errorf("error encoding to PNG: %v", err)
	}

	return buf.Bytes(), nil
}

func main() {
	flag.Parse()

	// Read config
	configData, err := os.ReadFile(*configFile)
	if err != nil {
		log.Fatalf("Failed to read config file: %v", err)
	}

	var config Config
	if err := yaml.Unmarshal(configData, &config); err != nil {
		log.Fatalf("Failed to parse config: %v", err)
	}

	// Set up Kafka consumer config
	consumerConfig := sarama.NewConfig()
	consumerConfig.Consumer.Group.Rebalance.Strategy = sarama.NewBalanceStrategyRoundRobin()
	consumerConfig.Consumer.Offsets.Initial = sarama.OffsetOldest
	consumerConfig.Consumer.Return.Errors = true
	consumerConfig.Version = sarama.V2_0_0_0

	// Create consumer group
	group, err := sarama.NewConsumerGroup([]string{*brokers}, "image-processor-group", consumerConfig)
	if err != nil {
		log.Fatalf("Error creating consumer group: %v", err)
	}
	defer group.Close()

	// Set up producer
	producerConfig := sarama.NewConfig()
	producerConfig.Producer.Return.Successes = true
	producerConfig.Producer.RequiredAcks = sarama.WaitForAll
	producerConfig.Producer.Retry.Max = 5
	producerConfig.Producer.MaxMessageBytes = 64 * 1024 * 1024 // Set max message size to 64MB

	producer, err := sarama.NewSyncProducer([]string{*brokers}, producerConfig)
	if err != nil {
		log.Fatalf("Failed to create producer: %v", err)
	}
	defer producer.Close()

	// Set up context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create consumer handler
	handler := NewConsumerGroupHandler(
		producer,
		config.Stages.DownloadImages.Producer,
		config.KafkaTopics[config.Stages.DownloadImages.Producer].Partitions,
		*doBenchmark,
	)

	// Handle OS signals
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	// Start metrics reporter
	wg := &sync.WaitGroup{}
	wg.Add(2) // One for consumer, one for metrics reporter

	go func() {
		defer wg.Done()
		ticker := time.NewTicker(metricsInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				currentMbps, currentImgRate, avgSpeed, avgImgRate, totalImages, totalGB, failedDownloads, runningTime := metrics.UpdateAndGetStats()

				log.Printf("\n=== Download Metrics ===\n"+
					"Current Speed: %.2f Mbps (%.2f images/sec)\n"+
					"Average Speed: %.2f Mbps (%.2f images/sec)\n"+
					"Total Images Downloaded: %d\n"+
					"Total Data Downloaded: %.2f GB\n"+
					"Failed Downloads: %d\n"+
					"Running Time: %v\n",
					currentMbps, currentImgRate,
					avgSpeed, avgImgRate,
					totalImages,
					totalGB,
					failedDownloads,
					runningTime.Round(time.Second))
			}
		}
	}()

	// Start consuming in a separate goroutine
	topics := []string{config.Stages.DownloadImages.Consumer}
	go func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():
				return
			default:
				if err := group.Consume(ctx, topics, handler); err != nil {
					log.Printf("Error from consumer: %v", err)
				}
			}
		}
	}()

	<-signals
	cancel()
	wg.Wait()
}
