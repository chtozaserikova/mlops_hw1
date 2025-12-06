#!/bin/bash
echo "ОСТАНОВКА ВСЕХ СЕРВИСОВ..."
kubectl delete -f k8s/ 2>/dev/null
docker compose down
echo "Все сервисы остановлены."