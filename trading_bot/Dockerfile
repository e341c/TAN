FROM golang:1.25-alpine AS builder

WORKDIR /src
COPY trading_bot/go.mod ./
RUN go mod download
COPY trading_bot/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /bot ./cmd/bot

FROM alpine:3.20
RUN apk add --no-cache ca-certificates curl
COPY --from=builder /bot /usr/local/bin/trading-bot
WORKDIR /app
EXPOSE 8081
ENTRYPOINT ["/usr/local/bin/trading-bot"]
