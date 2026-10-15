package exec

import (
	"context"
	"infer/signal"
)

type SignalSource interface {
	Signals(ctx context.Context) (<-chan signal.Signal, error)
}
