from django.apps import AppConfig
import os
import joblib


class FraudAppConfig(AppConfig):

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fraud_app'

    def ready(self):

        from . import services

        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        model_path = os.path.join(
            BASE_DIR,
            'ml_core',
            'models',
            'fraud_model.joblib'
        )

        scaler_path = os.path.join(
            BASE_DIR,
            'ml_core',
            'models',
            'scaler.joblib'
        )

        services.MODEL = joblib.load(model_path)
        services.SCALER = joblib.load(scaler_path)

        print('[OK] Fraud model loaded successfully')