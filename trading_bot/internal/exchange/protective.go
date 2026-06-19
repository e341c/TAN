package exchange

import "fmt"

func validProtectiveLevels(side Side, entry, sl, tp float64) error {
	if side == SideBuy {
		if sl > 0 && sl >= entry {
			return fmt.Errorf("long stop-loss %.4f must be below entry %.4f", sl, entry)
		}
		if tp > 0 && tp <= entry {
			return fmt.Errorf("long take-profit %.4f must be above entry %.4f", tp, entry)
		}
	} else {
		if sl > 0 && sl <= entry {
			return fmt.Errorf("short stop-loss %.4f must be above entry %.4f", sl, entry)
		}
		if tp > 0 && tp >= entry {
			return fmt.Errorf("short take-profit %.4f must be below entry %.4f", tp, entry)
		}
	}
	return nil
}

func formatPrice(p float64) string {
	return fmt.Sprintf("%.8f", p)
}
