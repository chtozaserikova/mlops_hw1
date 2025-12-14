import grpc
from concurrent import futures
import app_pb2
import app_pb2_grpc
import logging
import os
from logging.handlers import RotatingFileHandler
from flask import Flask
from typing import Any, Dict, List, Optional
from models import db, MLModel, AVAILABLE_MODELS, create_model_record, train_and_log_model, load_model_from_mlflow, convert_params

# Настройка логгера для gRPC сервера
logger = logging.getLogger('grpc_server')
logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)
file_handler = RotatingFileHandler(
    'logs/grpc_server.log', 
    maxBytes=1024 * 1024,
    backupCount=10
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db.init_app(app)

class MLService(app_pb2_grpc.MLServiceServicer):
    
    def HealthCheck(self, request: Any, context: Any) -> app_pb2.HealthResponse:
        logger.info("Health check requested via gRPC")
        return app_pb2.HealthResponse(status="ok")
    
    def GetModelClasses(self, request: Any, context: Any) -> app_pb2.ModelClassesResponse:
        logger.info("Request for available model classes via gRPC")
        model_classes = {}
        for key, val in AVAILABLE_MODELS.items():
            model_classes[key] = app_pb2.ModelClassInfo(
                class_name=val["class"].__name__,
                hyperparameters=val["hyperparameters"],
                description=val["description"]
            )
        logger.info(f"Returning {len(model_classes)} model classes via gRPC")
        return app_pb2.ModelClassesResponse(model_classes=model_classes)
    
    def ListModels(self, request: Any, context: Any) -> app_pb2.ListModelsResponse:
        logger.info("Request for list of all models via gRPC")
        with app.app_context():
            models = MLModel.query.all()
            model_list = []
            for m in models:
                model_response = app_pb2.ModelResponse(
                    id=str(m.id),
                    model_type=str(m.model_type),
                    params={str(k): str(v) for k, v in m.params.items()} if m.params else {},
                    created_at=m.created_at.isoformat() if m.created_at else "",
                    metrics={str(k): float(v) for k, v in m.metrics.items()} if m.metrics else {}
                )
                model_list.append(model_response)
            logger.info(f"Returning {len(model_list)} models via gRPC")
            return app_pb2.ListModelsResponse(models=model_list)
    
    def TrainModel(self, request: Any, context: Any) -> app_pb2.TrainResponse:
        logger.info("Starting model training request via gRPC")
        with app.app_context():
            model_type = request.model_type
            params = dict(request.params)
            X = [list(row.features) for row in request.X]
            y = list(request.y)
            logger.info(f"Training model type: {model_type} with {len(X)} samples via gRPC")
            if model_type not in AVAILABLE_MODELS:
                logger.error(f"Unsupported model type via gRPC: {model_type}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported model type")
            try:
                run_id, metrics, model_uri = train_and_log_model(model_type, params, X, y)
                record = create_model_record(run_id, model_type, params, model_uri, metrics)
                db.session.add(record)
                db.session.commit()
                logger.info(f"Model trained successfully via gRPC. ID: {run_id}, Metrics: {metrics}")
                return app_pb2.TrainResponse(
                    model_id=run_id, 
                    metrics={k: float(v) for k, v in metrics.items()}
                )
            except Exception as e:
                logger.error(f"Training failed via gRPC: {e}")
                context.abort(grpc.StatusCode.INTERNAL, f"Training failed: {str(e)}")

    def GetModel(self, request: Any, context: Any) -> app_pb2.ModelResponse:
        logger.info(f"Request for model info via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            logger.info(f"Returning model info via gRPC: {request.model_id}")
            return app_pb2.ModelResponse(
                id=str(record.id),
                model_type=str(record.model_type),
                params={str(k): str(v) for k, v in record.params.items()} if record.params else {},
                created_at=record.created_at.isoformat() if record.created_at else "",
                metrics={str(k): float(v) for k, v in record.metrics.items()} if record.metrics else {}
            )
    
    def DeleteModel(self, request: Any, context: Any) -> app_pb2.DeleteResponse:
        logger.info(f"Request to delete model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for deletion via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            # Мы НЕ удаляем артефакты из S3/MLflow, только запись из локальной БД
            db.session.delete(record)
            db.session.commit()
            
            logger.info(f"Model deleted successfully via gRPC: {request.model_id}")
            return app_pb2.DeleteResponse(success=True)

    def Predict(self, request: Any, context: Any) -> app_pb2.PredictResponse:
        logger.info(f"Prediction request for model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for prediction via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            try:
                model = load_model_from_mlflow(record.file_path)
                X = [list(row.features) for row in request.X]
                logger.info(f"Making prediction via gRPC with {len(X)} samples")
                preds = model.predict(X).tolist()
                logger.info(f"Prediction completed via gRPC. Returning {len(preds)} predictions")
                return app_pb2.PredictResponse(predictions=[float(p) for p in preds])
            except Exception as e:
                logger.error(f"Prediction failed via gRPC: {e}")
                context.abort(grpc.StatusCode.INTERNAL, "Failed to load model or make prediction")
    
    def RetrainModel(self, request: Any, context: Any) -> app_pb2.RetrainResponse:
        logger.info(f"Retrain request for model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for retraining via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            X = [list(row.features) for row in request.X]
            y = list(request.y)
            params = record.params
            model_type = record.model_type
            try:
                new_run_id, metrics, new_model_uri = train_and_log_model(model_type, params, X, y)
                db.session.delete(record)
                new_record = create_model_record(new_run_id, model_type, params, new_model_uri, metrics)
                db.session.add(new_record)
                db.session.commit()
                logger.info(f"Model retrained successfully via gRPC. Old ID: {request.model_id}, New ID: {new_run_id}")
                return app_pb2.RetrainResponse(
                    metrics={k: float(v) for k, v in metrics.items()}
                )
            except Exception as e:
                logger.error(f"Retraining failed via gRPC: {e}")
                context.abort(grpc.StatusCode.INTERNAL, f"Retraining failed: {str(e)}")

    def GetMetrics(self, request: Any, context: Any) -> app_pb2.MetricsResponse:
        logger.info(f"Metrics request for model via gRPC: {request.model_id}")
        with app.app_context():
            record = MLModel.query.filter_by(id=request.model_id).first()
            if not record:
                logger.warning(f"Model not found for metrics via gRPC: {request.model_id}")
                context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            
            logger.info(f"Returning metrics via gRPC for model: {request.model_id}")
            return app_pb2.MetricsResponse(
                metrics={str(k): float(v) for k, v in record.metrics.items()} if record.metrics else {}
            )

def serve() -> None:
    logger.info("Starting gRPC server")
    with app.app_context():
        db.create_all()
        logger.info("Database tables created for gRPC server")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    app_pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("gRPC server started on port 50051")
    print("gRPC server started on port 50051")
    server.start()
    logger.info("gRPC server waiting for termination")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()