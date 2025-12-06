import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Union, Optional, Tuple
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
# from dotenv import load_dotenv

# load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
# mlflow.set_experiment("mlops_homework_experiment")
mlflow.set_experiment("final_check_experiment")

logger = logging.getLogger('models')
logger.setLevel(logging.INFO)

db = SQLAlchemy()

class MLModel(db.Model):
    id = db.Column(db.String, primary_key=True) 
    model_type = db.Column(db.String(120))
    params = db.Column(db.JSON)
    file_path = db.Column(db.String(500)) 
    created_at = db.Column(db.DateTime)
    metrics = db.Column(db.JSON)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует модель в словарь для API ответов"""
        logger.debug(f"Converting model {self.id} to dictionary")
        return {
            'id': self.id,
            'model_type': self.model_type,
            'params': self.params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metrics': self.metrics
        }

# Доступные модели
AVAILABLE_MODELS = {
    'random_forest': {
        'class': RandomForestClassifier,
        'hyperparameters': ['n_estimators', 'max_depth', 'random_state'],
        'description': 'Random Forest Classifier'
    },
    'logistic_regression': {
        'class': LogisticRegression,
        'hyperparameters': ['C', 'solver', 'max_iter'],
        'description': 'Logistic Regression'
    }
}

def convert_params(params: Dict[str, Any]) -> Dict[str, Union[int, float, str]]:
    """Конвертирует строковые параметры в правильные типы"""
    logger.debug(f"Converting parameters: {params}")
    converted_params = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value.isdigit():
                converted_params[key] = int(value)
                logger.debug(f"Converted parameter {key} to int: {value}")
            else:
                try:
                    converted_params[key] = float(value)
                    logger.debug(f"Converted parameter {key} to float: {value}")
                except ValueError:
                    converted_params[key] = value
                    logger.debug(f"Parameter {key} kept as string: {value}")
        else:
            converted_params[key] = value
            logger.debug(f"Parameter {key} kept as original type: {type(value)}")
    
    logger.info(f"Parameters conversion completed. Converted {len(converted_params)} parameters")
    return converted_params

def calculate_metrics(y_true: List[Union[int, float]], y_pred: List[Union[int, float]]) -> Dict[str, float]:
    """Вычисляет метрики модели"""
    logger.debug(f"Calculating metrics for {len(y_true)} samples")
    try:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }
        
        logger.info(f"Metrics calculated - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        # Возвращаем метрики по умолчанию в случае ошибки
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }

def train_and_log_model(
    model_type: str, 
    params: Dict[str, Any], 
    X: List[List[float]], 
    y: List[Union[int, float]]
) -> Tuple[str, Dict[str, float], str]:
    """
    Обучает модель и логирует её в MLflow.
    Возвращает кортеж: (run_id, metrics, model_uri)
    """
    if mlflow.active_run():  
        logger.warning("Found active MLflow run. Ending it forcefully.") 
        mlflow.end_run()    
    logger.info(f"Starting training for {model_type}")
    converted_params = convert_params(params)
    ModelClass = AVAILABLE_MODELS[model_type]['class']
    model = ModelClass(**converted_params)
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)
    # Логирование в MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(converted_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Model logged to MLflow. Run ID: {run_id}")
    return run_id, metrics, model_uri

def load_model_from_mlflow(model_uri: str) -> Any:
    """Загружает модель из MLflow (S3)"""
    logger.info(f"Loading model from URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model

def create_model_record(
    model_id: str, 
    model_type: str, 
    params: Dict[str, Any], 
    file_path: str, 
    metrics: Dict[str, float]
) -> MLModel:
    """Создает запись модели в БД"""
    logger.info(f"Creating model record: ID={model_id}, Type={model_type}")
    logger.debug(f"Model params: {params}, Metrics: {metrics}")
    
    record = MLModel(
        id=model_id,
        model_type=model_type,
        params=params,
        file_path=file_path, 
        created_at=datetime.now(),
        metrics=metrics
    )
    logger.debug(f"Model record created successfully: {model_id}")
    return record