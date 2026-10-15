package signal

import "context"

type Publisher interface {
	PublishSignal(ctx context.Context, s Signal) error 
}