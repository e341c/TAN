package config

type Kafka struct {
	// The list of brokers used to discover the partitions available on the
	// kafka cluster.
	//
	// This field is required, attempting to create a writer with an empty list
	// of brokers will panic.
	Brokers []string

	// The topic that the writer will produce messages to.
	//
	// If provided, this will be used to set the topic for all produced messages.
	// If not provided, each Message must specify a topic for itself. This must be
	// mutually exclusive, otherwise the Writer will return an error.
	Topic string

	// GroupID holds the optional consumer group id.  If GroupID is specified, then
	// Partition should NOT be specified e.g. 0
	GroupID string
}

type Generator struct {
	EnterLong, ExitLong   float64 // например 0.58 / 0.52
	EnterShort, ExitShort float64 // например 0.42 / 0.48
	TargetVol             float64 // целевая волатильность позиции (в тех же единицах, что Sigma)
	MaxWeight             float64 // потолок доли капитала на инструмент
}

type Config struct {
	Kafka     Kafka
	Generator Generator
}

func Load() (*Config, error) {
	return nil, nil
}
