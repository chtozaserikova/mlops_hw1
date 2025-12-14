DOCKER_USERNAME ?= tvoy_login
IMAGE_NAME = my_ml_service

.PHONY: build-push test lint

build-push:
	# Тут выведется логин, который используется
	@echo "Using Docker User: $(DOCKER_USERNAME)"
	docker login
	docker build -t $(DOCKER_USERNAME)/$(IMAGE_NAME):latest .
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):latest

# запуск тестов
test:
	MLFLOW_TRACKING_URI=file:./test_mlruns pytest tests/ -v

# проверка синтаксиса
lint:
	pylint --disable=R,C,E0401,E1101,W1203,W0718,W0611 app.py models.py grpc_server.py || true
# Мы отключили за неинформативность:
# R, C  - рекомендации стиля и сложности
# E0401 - ошибки импорта (т.к. запускаем без venv)
# E1101 - ошибки доступа к полям (т.к. Pylint не понимает Protobuf)
# W1203 - придирки к f-строкам в логгере
# W0718 - широкие except (ловим все ошибки)
# W0611 - неиспользуемые импорты