echo "=================================================="
echo "ЗАПУСК В РЕЖИМЕ DOCKER COMPOSE"
echo "=================================================="

echo ">>> Останавливаем старые контейнеры..."
docker compose down

echo ">>> Запускаем сервисы..."
docker compose up --build -d

echo "=================================================="
echo "ГОТОВО! Сервисы доступны по адресам:"
echo "   Dashboard: http://localhost:8501"
echo "   Flask API: http://localhost:5000"
echo "   MLflow UI: http://localhost:5002"
echo "   MinIO:     http://localhost:9001"
echo "=================================================="