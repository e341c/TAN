package exchange

import "testing"

func TestValidProtectiveLevelsLong(t *testing.T) {
	if err := validProtectiveLevels(SideBuy, 100, 90, 110); err != nil {
		t.Fatalf("valid long levels: %v", err)
	}
	if err := validProtectiveLevels(SideBuy, 100, 100, 110); err == nil {
		t.Fatal("expected error when SL >= entry")
	}
	if err := validProtectiveLevels(SideBuy, 100, 90, 100); err == nil {
		t.Fatal("expected error when TP <= entry")
	}
}

func TestValidProtectiveLevelsShort(t *testing.T) {
	if err := validProtectiveLevels(SideSell, 100, 110, 90); err != nil {
		t.Fatalf("valid short levels: %v", err)
	}
	if err := validProtectiveLevels(SideSell, 100, 90, 80); err == nil {
		t.Fatal("expected error when short SL <= entry")
	}
}
