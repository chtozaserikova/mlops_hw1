#!/bin/bash
export PATH=$PATH:"/c/Program Files/Kubernetes/Minikube"
echo "=================================================="
echo "ЗАПУСК В РЕЖИМЕ KUBERNETES"
echo "=================================================="

echo ">>> [1/6] Очистка..."
docker compose down
kubectl delete -f k8s/ 2>/dev/null || true

echo ">>> [2/6] Поднимаем инфраструктуру (MinIO, MLflow)..."
docker compose up -d minio db mlflow create_buckets

echo ">>> [3/6] Проверка Minikube..."
minikube start

echo ">>> [4/6] Сборка Docker-образа..."
docker build -t my_ml_service:latest .

echo ">>> [5/6] Загрузка образа в Minikube..."
minikube image load my_ml_service:latest

# 6. Деплой приложений
echo ">>> [6/6] Деплой Flask и gRPC в Kubernetes..."
kubectl apply -f k8s/

echo "=================================================="
echo "⏳ Ждем запуска Pods (15 сек)..."
sleep 15
kubectl get pods

echo ""
echo "=================================================="
echo "ГОТОВО!"
echo "   Инфраструктура работает в Docker Compose (localhost)"
echo "   Приложение работает в Kubernetes"
echo ""
echo "Ссылка на Swagger API:"
minikube service ml-flask-service --url
echo "=================================================="