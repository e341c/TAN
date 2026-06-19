package orders

import "context"

// Writer persists trading order records. Implemented by Store; nil-safe via bot layer.
type Writer interface {
	Insert(ctx context.Context, r Record) error
}
