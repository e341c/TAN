package kafka

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"time"

	"infer/config"
	"infer/signal"

	"github.com/segmentio/kafka-go"
)

type Consumer struct {
	reader *kafka.Reader
}

func NewConsumer(cfg config.Kafka) *Consumer {
	return &Consumer{
		reader: kafka.NewReader(kafka.ReaderConfig{
			Brokers:     cfg.Brokers,
			Topic:       cfg.Topic,
			GroupID:     cfg.GroupID,
			MinBytes:    1,
			MaxBytes:    10e6,
			MaxWait:     50 * time.Millisecond,
			StartOffset: kafka.LastOffset, // действует только для новой группы
		}),
	}
}

func (c *Consumer) Fetch(ctx context.Context) (kafka.Message, error) {
	return c.reader.FetchMessage(ctx)
}

func (c *Consumer) Commit(ctx context.Context, m kafka.Message) error {
	return c.reader.CommitMessages(ctx, m)
}

func (c *Consumer) Close() error { return c.reader.Close() }

// SignalConsumer реализует exec.SignalSource.
type SignalConsumer struct {
	c      *Consumer
	log    *slog.Logger
	maxAge time.Duration
}

func NewSignalConsumer(c *Consumer, maxAge time.Duration) *SignalConsumer {
	return &SignalConsumer{c: c, maxAge: maxAge}
}

func (s *SignalConsumer) Run(ctx context.Context, handle func(context.Context, signal.Signal) error) error {
	for {
		m, err := s.c.Fetch(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, kafka.ErrGroupClosed) {
				return nil
			}
			return err
		}

		var sig signal.Signal
		switch {
		case json.Unmarshal(m.Value, &sig) != nil:
			// poison message: логируем и пропускаем, иначе застрянем на нём навсегда
			s.log.Error("bad signal", "partition", m.Partition, "offset", m.Offset)
		case time.Since(sig.BarTime) > s.maxAge:
			// протухший сигнал (лаг, рестарт): исполнять нельзя
			s.log.Warn("stale signal dropped", "id", sig.ID)
		default:
			if err := handle(ctx, sig); err != nil {
				return err // не коммитим -> после рестарта сообщение придёт снова
			}
		}

		if err := s.c.Commit(ctx, m); err != nil {
			return err
		}
	}
}
