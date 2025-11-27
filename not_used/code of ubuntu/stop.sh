echo "🛑 Stopping Docker containers..."
cd ~/bigdata-stack && docker-compose down
echo "🧠 Stopping Zeppelin..."
sudo /opt/zeppelin/bin/zeppelin-daemon.sh stop
echo "✅ Everything stopped cleanly."








