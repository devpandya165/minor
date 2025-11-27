#!/bin/bash
# ==============================================
# 🚀 COMPLETE BIG DATA PIPELINE STARTUP SCRIPT
# ==============================================
# Author: Dev Pandya
# Description: Starts Docker containers (Kafka, Zookeeper.),
# activates venv, launches Zeppelin, and runs electricity data producer.

echo " Changing directory to project folder..."
cd ~/bigdata-stack || { echo " Folder ~/bigdata-stack not found!"; exit 1; }

echo " Starting Docker containers (Kafka, Zookeeper)..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo " Docker failed to start. Check docker-compose.yml"
    exit 1
fi

echo "✅ Docker containers running:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# ---------------------------------------------------------------
# Activate Python virtual environment
# ---------------------------------------------------------------
if [ -d "zeppelin_venv" ]; then
    echo "🐍 Activating virtual environment..."
    source zeppelin_venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv zeppelin_venv
    source zeppelin_venv/bin/activate
    pip install --upgrade pip
    pip install pandas numpy joblib pyspark kafka-python lightgbm
fi

# ---------------------------------------------------------------
# Start Apache Zeppelin
# ---------------------------------------------------------------
echo " Starting Apache Zeppelin..."
sudo /opt/zeppelin/bin/zeppelin-daemon.sh start

# Wait for Zeppelin to initialize
sleep 10
echo "✅ Zeppelin should now be live at: http://localhost:8080"

# ---------------------------------------------------------------
# Check if models folder exists
# ---------------------------------------------------------------
if [ -d "/home/devpandya/models" ]; then
    echo "📁 Models folder found"
else
    echo "⚠️ Models folder not found at /home/devpandya/models"

    echo "   You can add models later when needed."
fi


# Wait until Kafka broker is fully ready
echo " Checking Kafka broker readiness..."
for i in {1..10}; do
    nc -z localhost 9092 && echo "✅ Kafka broker is ready!" && break
    echo "🔄 Waiting for Kafka broker... ($i/10)"
    sleep 3
done

# If Kafka still isn't ready, exit
if ! nc -z localhost 9092; then
    echo "❌ Kafka broker is not reachable on port 9092. Aborting."
    exit 1
fi

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
echo ""
echo "All systems started successfully!"
echo "👉 Zeppelin: http://localhost:8080"
echo "👉 Kafka topic: electricity_topic"
