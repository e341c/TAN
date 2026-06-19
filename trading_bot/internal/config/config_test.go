package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadExample(t *testing.T) {
	path := filepath.Join("..", "..", "config", "strategy.example.yaml")
	if _, err := os.Stat(path); err != nil {
		t.Skip("example config not present yet")
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Bot.Mode != "paper" {
		t.Fatalf("expected paper mode, got %s", cfg.Bot.Mode)
	}
}

func TestValidateLiveRequiresKeys(t *testing.T) {
	if err := ValidateLiveCredentials("live", "", ""); err == nil {
		t.Fatal("expected error for missing keys")
	}
	if err := ValidateLiveCredentials("paper", "", ""); err != nil {
		t.Fatalf("paper should not require keys: %v", err)
	}
}
