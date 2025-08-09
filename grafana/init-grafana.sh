#!/bin/bash

echo "=== Grafana Initialization Script ==="
echo "Waiting for Prometheus to be ready..."

# Wait for Prometheus to be ready
PROMETHEUS_URL="http://prometheus:9090"
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: Checking Prometheus at $PROMETHEUS_URL"
    
    if wget --spider --quiet --timeout=5 "$PROMETHEUS_URL/-/ready" 2>/dev/null; then
        echo "✅ Prometheus is ready!"
        break
    elif wget --spider --quiet --timeout=5 "$PROMETHEUS_URL/api/v1/status/config" 2>/dev/null; then
        echo "✅ Prometheus is ready (alternative check)!"
        break
    else
        echo "⏳ Prometheus not ready yet, waiting 10 seconds..."
        sleep 10
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo "❌ Prometheus failed to become ready after $MAX_ATTEMPTS attempts"
    echo "🔄 Starting Grafana anyway..."
else
    echo "🚀 Prometheus is ready, starting Grafana..."
fi

# Test datasource connectivity
echo "Testing datasource connectivity..."
if wget --spider --quiet --timeout=5 "$PROMETHEUS_URL/api/v1/query?query=up" 2>/dev/null; then
    echo "✅ Prometheus API is responding"
else
    echo "⚠️ Prometheus API test failed, but continuing..."
fi

echo "=== Starting Grafana Server ==="
exec /run.sh "$@"
