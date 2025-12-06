import os
import logging
from logging.handlers import RotatingFileHandler
    
from flask import Flask, redirect, url_for, session, request, Response
from flask_restx import Api, Resource, Namespace, fields, abort
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix
from models import db, MLModel, AVAILABLE_MODELS, create_model_record, train_and_log_model, load_model_from_mlflow
from typing import Dict, Any, List, Tuple, Union, Optional

"""
Setting app configurations
"""

# Настройка логгера для Flask приложения
logger = logging.getLogger('flask_app')
logger.setLevel(logging.INFO)

# Файловый обработчик
os.makedirs("logs", exist_ok=True)
file_handler = RotatingFileHandler(
    'logs/flask_api.log', 
    maxBytes=1024 * 1024,  
    backupCount=10
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Инициализация базы данных
db.init_app(app)

api = Api(
    app, 
    title='REST API models depl',
    description="""Documentation for models delp.\n\n
    Before getting to work with this application please go through authorization procedure.
    To do so add /login to your current url.
    (for example, http://localhost:5000/login)"""
 )

namespace = api.namespace('', 'Click the down arrow to expand the content')

"""
Initializing authorization via github
"""

oauth = OAuth(app)
github = oauth.register(
    name='github',
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

@app.route("/") 
def index() -> Response: 
    logger.info("Redirecting from index to login page")
    return redirect("/login")

@app.route('/login')
def registro() -> Response:
    logger.info("Starting OAuth login process")
    github = oauth.create_client('github')
    redirect_uri = url_for('authorize', _external=True)
    logger.info(f"Redirecting to GitHub OAuth: {redirect_uri}")
    return github.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize() -> Response:
    logger.info("Processing OAuth authorization callback")
    github = oauth.create_client('github')
    token = github.authorize_access_token()
    resp = github.get('user', token=token)
    profile = resp.json()
    
    if 'id' not in profile:
        logger.error("GitHub authorization failed - no user ID in profile")
        abort(400, "GitHub authorization failed")
        
    github_id = profile['id']
    session['token_oauth'] = token
    session['github_id'] = profile['id']
    logger.info(f"Successful GitHub authorization for user ID: {github_id}")
    return redirect(url_for('index'))

def get_user_id() -> Optional[int]:
    # вернет ID пользователя или None, если пользователь не авторизован
    if 'github_id' in session:
        return session['github_id']
    return 

# Swagger models
train_model = api.model('TrainModel', {
    'model_type': fields.String(required=True, description='Model type (random_forest / logistic_regression)'),
    'params': fields.Raw(required=True, description='Model parameters'),
    'X': fields.List(fields.List(fields.Float), required=True, description='Features'),
    'y': fields.List(fields.Integer, required=True, description='Labels')
})

predict_model = api.model('PredictModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='Data for making predictions')
})

retrain_model = api.model('RetrainModel', {
    'X': fields.List(fields.List(fields.Float), required=True, description='Features'),
    'y': fields.List(fields.Integer, required=True, description='Labels')
})

# Endpoints

@namespace.route('/health')
class Health(Resource):
    @api.doc(description="Проверка статуса сервиса")
    def get(self) -> Tuple[Dict[str, str], int]:
        logger.info("Health check requested")
        return {'status': 'ok'}, 200


@namespace.route('/model-classes')
class ModelClasses(Resource):
    @api.doc(description="List of models available and their parameters")
    def get(self) -> Tuple[Dict[str, Any], int]:
        logger.info("Request for available model classes")
        models_info = {}
        for key, val in AVAILABLE_MODELS.items():
            models_info[key] = {
                "class_name": val["class"].__name__,
                "hyperparameters": val["hyperparameters"],
                "description": val["description"]
            }
        logger.info(f"Returning {len(models_info)} model classes")
        return models_info, 200

@namespace.route('/models/train')
class TrainModel(Resource):
    @api.doc(description="Model training")
    @api.expect(train_model)
    def post(self) -> Tuple[Dict[str, Any], int]:
        logger.info("Starting model training request")
        data = request.get_json()
        model_type = data.get('model_type')
        params = data.get('params', {})
        X = data.get('X')
        y = data.get('y')

        logger.info(f"Training model type: {model_type} with {len(X)} samples")

        if model_type not in AVAILABLE_MODELS:
            logger.error(f"Unsupported model type: {model_type}")
            abort(400, 'Unsupported model type')

        try:
            # Обучаем, считаем метрики и логируем в MLflow 
            run_id, metrics, model_uri = train_and_log_model(model_type, params, X, y)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            abort(500, f"Training failed: {str(e)}")

        record = create_model_record(run_id, model_type, params, model_uri, metrics)
        db.session.add(record)
        db.session.commit()

        logger.info(f"Model trained successfully. ID: {run_id}, Metrics: {metrics}")
        return {'model_id': run_id, 'metrics': metrics}, 201


@namespace.route('/models')
class ListModels(Resource):
    @api.doc(description="Get list of models trained")
    def get(self) -> Tuple[List[Dict[str, Any]], int]:
        logger.info("Request for list of all models")
        models = MLModel.query.all()
        result = [model.to_dict() for model in models]
        logger.info(f"Returning {len(result)} models")
        return result, 200


@namespace.route('/models/<string:model_id>')
class ModelById(Resource):
    @api.doc(description="Get information on a trained model")
    def get(self, model_id: str) -> Tuple[Dict[str, Any], int]:
        logger.info(f"Request for model info: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found: {model_id}")
            abort(404, 'Model not found')
        logger.info(f"Returning model info: {model_id}")
        return record.to_dict(), 200

    @api.doc(description="Delete model")
    def delete(self, model_id: str) -> Tuple[str, int]:
        logger.info(f"Request to delete model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for deletion: {model_id}")
            abort(404, 'Model not found')
        
        db.session.delete(record)
        db.session.commit()
        
        logger.info(f"Model record deleted successfully: {model_id}")
        return '', 204


@namespace.route('/models/<string:model_id>/predict')
class ModelPredict(Resource):
    @api.doc(description="Make prediction")
    @api.expect(predict_model)
    def post(self, model_id: str) -> Tuple[Dict[str, List[float]], int]:
        logger.info(f"Prediction request for model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for prediction: {model_id}")
            abort(404, 'Model not found')
        try:
            # Загружаем модель 
            model = load_model_from_mlflow(record.file_path)
            
            X = request.json.get('X')
            logger.info(f"Making prediction with {len(X)} samples")
            
            preds = model.predict(X).tolist()
            logger.info(f"Prediction completed. Returning {len(preds)} predictions")
            return {'predictions': preds}, 200
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            abort(500, "Failed to load model or make prediction")


@namespace.route('/models/<string:model_id>/retrain')
class ModelRetrain(Resource):
    @api.doc(description="Retrain existing model")
    @api.expect(retrain_model)
    def post(self, model_id: str) -> Tuple[Dict[str, Any], int]:
        logger.info(f"Retrain request for model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for retraining: {model_id}")
            abort(404, 'Model not found')
        X = request.json.get('X')
        y = request.json.get('y')
        params = record.params
        model_type = record.model_type
        try:
            new_run_id, metrics, new_model_uri = train_and_log_model(model_type, params, X, y)
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            abort(500, f"Retraining failed: {str(e)}")
        db.session.delete(record)
        new_record = create_model_record(new_run_id, model_type, params, new_model_uri, metrics)
        db.session.add(new_record)
        db.session.commit()
        logger.info(f"Model retrained. Old ID: {model_id}, New ID: {new_run_id}, Metrics: {metrics}")
        return {
            'status': 'retrained', 
            'new_model_id': new_run_id, 
            'metrics': metrics
        }, 200


@namespace.route('/metrics/<string:model_id>')
class ModelMetrics(Resource):
    @api.doc(description="Get model scores")
    def get(self, model_id: str) -> Tuple[Dict[str, float], int]:
        logger.info(f"Metrics request for model: {model_id}")
        record = MLModel.query.filter_by(id=model_id).first()
        if not record:
            logger.warning(f"Model not found for metrics: {model_id}")
            abort(404, 'Model not found')
        logger.info(f"Returning metrics for model: {model_id}")
        return record.metrics, 200   

if __name__ == '__main__':
    logger.info("Starting Flask application")
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")
    logger.info("Flask app running in debug mode")
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)