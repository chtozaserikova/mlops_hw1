IMAGE_NAME = my_ml_service:latest

k8s-start: build load deploy info

build:
	docker build -t $(IMAGE_NAME) .
load:
	@echo "Loading image into Minikube... "
	minikube image load $(IMAGE_NAME)
deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/flask-deployment.yaml
	kubectl apply -f k8s/flask-service.yaml
	kubectl apply -f k8s/grpc-deployment.yaml
	kubectl apply -f k8s/grpc-service.yaml
info:
	@echo "Waiting for pods to start..."
	@sleep 10
	kubectl get pods
	@echo ""
	@echo "==================================================="
	@echo "Flask API is running at:"
	@minikube service ml-flask-service --url
	@echo ""
	@echo "gRPC Server is running on port 30051"
	@echo "==================================================="

k8s-stop:
	kubectl delete -f k8s/