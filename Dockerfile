FROM golang:1.25.0 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 go build -o /app/commandcode-proxy ./cmd/proxy

FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  ca-certificates \
  curl \
  sqlite3 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/commandcode-proxy /app/commandcode-proxy

RUN mkdir -p /app/data

ENV PORT=3000
EXPOSE 3000

CMD ["/app/commandcode-proxy"]