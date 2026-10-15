package kafka

import (
	"context"
	"encoding/json"
	"infer/config"
	"infer/signal"

	"github.com/segmentio/kafka-go"
)

type Publisher struct {
	cfg    config.Kafka
	writer *kafka.Writer
}

func NewPublisher(cfg config.Kafka) *Publisher {
	return &Publisher{
		cfg: cfg,
		writer: kafka.NewWriter(kafka.WriterConfig{
			Brokers: cfg.Brokers,
			Topic:   cfg.Topic,
		}),
	}
}

func (p *Publisher) Publish(ctx context.Context, key string, value []byte) error {
	return p.writer.WriteMessages(ctx, kafka.Message{
		Key:   []byte(key),
		Value: value,
	})
}

func (p *Publisher) Close() error { return p.writer.Close() }

// SignalPublisher реализует signal.Publisher.
type SignalPublisher struct {
	pub *Publisher
}

func NewSignalPublisher(pub *Publisher) *SignalPublisher {
	return &SignalPublisher{pub: pub}
}

func (s *SignalPublisher) PublishSignal(ctx context.Context, sig signal.Signal) error {
	b, err := json.Marshal(sig) // позже можно заменить на protobuf
	if err != nil {
		return err
	}
	return s.pub.Publish(ctx, sig.Symbol, b) // ключ = символ
}
