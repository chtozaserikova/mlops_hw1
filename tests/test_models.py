import pytest
from unittest.mock import MagicMock, patch
from models import train_and_log_model, convert_params

# ==========================================
# ТЕСТ А: Unit-тест 
# ==========================================
def test_convert_params_logic():
    """
   Проверяем, что строки превращаются в числа, где это возможно.
    """
    # Входные данные
    input_params = {
        "n_estimators": "100", 
        "learning_rate": "0.05",
        "solver": "liblinear", 
        "random_state": 42 
    }
    # Ожидаемый результат
    expected_result = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "solver": "liblinear",
        "random_state": 42
    }
    real_result = convert_params(input_params)
    assert real_result == expected_result
    assert isinstance(real_result["n_estimators"], int)
    assert isinstance(real_result["learning_rate"], float)

# ==========================================
# ТЕСТ Б: Тест с Моком (Имитация S3/MLflow)
# ==========================================
@patch("models.mlflow") 
def test_train_and_log_model_mocked(mock_mlflow):
    """
    Тестируем основную функцию обучения.
    Аргумент mock_mlflow - это "кукла", которая заменит реальную библиотеку mlflow.
    """
    # Настройка Мока
    # Имитиурем поведение 'with mlflow.start_run() as run:'
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id_123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    
    # Подготовка данных и запуск
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    params = {"n_estimators": "10", "max_depth": "3"}
    model_type = "random_forest"
    run_id, metrics, model_uri = train_and_log_model(model_type, params, X, y)

    # Проверки
    # Проверяем, что функция вернула нужный ID
    assert run_id == "test_run_id_123"
    # Проверяем формат URI
    assert model_uri == "runs:/test_run_id_123/model"
    # Проверяем, что метрики посчитались
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    mock_mlflow.log_params.assert_called()       # Были ли записаны параметры?
    mock_mlflow.log_metrics.assert_called()      # Были ли записаны метрики?
    mock_mlflow.sklearn.log_model.assert_called() # Была ли попытка сохранить модель?