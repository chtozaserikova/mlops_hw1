#!/bin/bash

echo "Generating data..."
python prepare_data.py

# Инициализируем DVC 
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
fi

# Настраиваем удаленное хранилище
echo "Configuring DVC remote..."
dvc remote add -d minio s3://dvc-storage
# URL MinIO
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
dvc remote modify minio use_ssl false

# Добавляем данные под контроль DVC
echo "Adding data to DVC..."
dvc add data/train.csv

# Отправляем в MinIO
echo "Pushing data to MinIO..."
dvc push

echo "SUCCESS! DVC is configured and data is pushed to MinIO."