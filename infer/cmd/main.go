package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"golang.org/x/sync/errgroup"

	"infer/config"
	sig "infer/signal"
	k "infer/transport/kafka"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(log)

	if err := run(log); err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pub := k.NewPublisher(cfg.Kafka)
	defer pub.Close()
	signalPub := k.NewSignalPublisher(pub)

	gen := sig.NewGenerator(cfg.Generator)

	modelOut := make(chan sig.Input, 64) // TODO: заменить на выход пакета model
	auditCh := make(chan sig.Signal, 256)
	// riskIn := make(chan sig.Signal, 16) // горячий путь в risk -> exec

	g, ctx := errgroup.WithContext(ctx)

	// Горячий путь: модель -> сигнал -> risk. Здесь нельзя блокироваться на сети.
	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case in, ok := <-modelOut:
				if !ok {
					return nil
				}
				s := gen.Next(in)
				// riskIn <- s

				select {
				case auditCh <- s:
				default:
					log.Warn("audit buffer full, signal not published", "id", s.ID)
				}
			}
		}
	})

	// Публикация в Kafka для аудита и внешних подписчиков, отдельно от торговли.
	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case s := <-auditCh:
				if err := signalPub.PublishSignal(ctx, s); err != nil {
					log.Error("publish signal", "id", s.ID, "err", err)
				}
			}
		}
	})

	return g.Wait()
}
