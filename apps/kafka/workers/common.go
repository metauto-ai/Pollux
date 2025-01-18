package main

import (
	"fmt"
	"log"
	"os"

	"github.com/IBM/sarama"
	"gopkg.in/yaml.v3"
)

type ProducerConfig struct {
	Name       string `json:"name"`
	Partitions int    `json:"partitions"`
	Producer   sarama.AsyncProducer
}

type DownloadResult struct {
	URL     string
	Success bool
	Error   error
	Size    int
}

type MessageContent struct {
	DocumentIDs []string `json:"document_ids"`
	ImageURLs   []string `json:"image_urls"`
}

type Job struct {
	URL        string
	DocumentID string
}

type MessagePayload struct {
	DocumentIDs   []string `json:"document_ids"`
	ImageContents []string `json:"image_contents"`
}

// Structs to represent the YAML structure
type KafkaTopic struct {
	Name       string `yaml:"name"`
	Partitions int    `yaml:"partitions"`
}

type Stage struct {
	BatchSize    int      `yaml:"batch_size,omitempty"`
	Dataset      string   `yaml:"dataset,omitempty"`
	Consumer     string   `yaml:"consumer,omitempty"`
	ProducerList []string `yaml:"producer_list,omitempty"`
}

type Config struct {
	KafkaTopics map[string]KafkaTopic `yaml:"kafka_topics"`
	Stages      map[string]Stage      `yaml:"stages"`
}

func load_yaml_config(file_path string) Config {
	// Open the YAML file
	file, err := os.Open(file_path)
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	// Parse the YAML file
	var config Config
	decoder := yaml.NewDecoder(file)
	if err := decoder.Decode(&config); err != nil {
		log.Fatalf("Failed to decode YAML: %v", err)
	}

	return config
}

func humanReadableSize(bytes int) string {
	const (
		_  = iota // ignore first value by assigning to blank identifier
		KB = 1 << (10 * iota)
		MB
		GB
		TB
	)

	switch {
	case bytes < KB:
		return fmt.Sprintf("%d B", bytes)
	case bytes < MB:
		return fmt.Sprintf("%.2f KB", float64(bytes)/KB)
	case bytes < GB:
		return fmt.Sprintf("%.2f MB", float64(bytes)/MB)
	case bytes < TB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/GB)
	default:
		return fmt.Sprintf("%.2f TB", float64(bytes)/TB)
	}
}
