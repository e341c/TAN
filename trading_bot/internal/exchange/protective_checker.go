package exchange

type ProtectiveChecker interface {
	CheckProtectiveExit(price float64) (triggered bool, reason string)
}
