package state

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/eriktabys/tan-model/trading_bot/internal/exchange"
)

type BotState struct {
	LastTargetTS string             `json:"last_target_ts"`
	LastEntryAt  time.Time          `json:"last_entry_at"`
	TradeID      string             `json:"trade_id,omitempty"`
	Position     *exchange.Position `json:"position,omitempty"`
}

type Store struct {
	path string
}

func NewStore(path string) *Store {
	return &Store{path: path}
}

func (s *Store) Load() (*BotState, error) {
	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return &BotState{}, nil
		}
		return nil, fmt.Errorf("read state: %w", err)
	}
	var st BotState
	if err := json.Unmarshal(data, &st); err != nil {
		return nil, fmt.Errorf("decode state: %w", err)
	}
	return &st, nil
}

func (s *Store) Save(st *BotState) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("mkdir state dir: %w", err)
	}
	data, err := json.Marshal(st)
	if err != nil {
		return fmt.Errorf("encode state: %w", err)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write state temp: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("rename state: %w", err)
	}
	return nil
}
