"""
Ce fichier contient le script principal du projet.

Il utilise les modules préalablement définis pour nettoyer les données, 
effectuer un ingénierie des fonctionnalités, 
et entraîner des modèles d'apprentissage automatique.
"""

# Importation des modules nécessaires
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from scr.preprocessing.data_cleaning import DataCleaner
from scr.preprocessing.feature_engineering import DataFeatureEngineering
from scr.models.xgboost_model import XGBoostModel, XGBoostPipeline
from scr.models.baseline_model import RLModel, RLModelPipeline
from scr.config import DATA_FILE_PATH

def main():
    """
    Fonction principale pour l'exécution des étapes de nettoyage, d'ingénierie des features
    et d'entraînement des modèles.

    Returns:
        None
    """
    # Configuration du logging
    logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Chargement des données
    data_conso = pd.read_csv(DATA_FILE_PATH)

    # Nettoyage des données
    logging.info('Début du nettoyage des données.')
    cleaning_pipeline = DataCleaner()
    data_conso_clean = cleaning_pipeline.transform(data_conso)
    train_features, test_features, train_target, test_target = cleaning_pipeline.split_data(data_conso_clean)
    logging.info('Nettoyage des données terminé.')


    # Exécute le pipeline de feature engineering sur les données nettoyées
    logging.info('Début de l\'ingénierie des fonctionnalités.')
    feature_engineering_pipeline = DataFeatureEngineering()
    train_features, test_features, train_target, test_target = feature_engineering_pipeline.transform(
        train_features,
        test_features,
        train_target,
        test_target,
        )
    logging.info('Ingénierie des fonctionnalités terminée.')

    # Modèle de base (RL)
    logging.info("Début de l'entraînement du modèle de base (RL).")
    baseline_model = RLModel()
    baseline_pipeline = RLModelPipeline(baseline_model)
    baseline_pipeline.train_fit(train_features, test_features, train_target, test_target)
    logging.info('Fin de l\'entraînement du modèle de base (RL).')
    # Évaluation du modèle et enregistrement des métriques
    rmse_test, r2_test = baseline_model.get_score()
    logging.info('Métriques de test du modèle de base (RL):')
    logging.info('RMSE: %f', rmse_test)
    logging.info('R2: %f', r2_test)
    # Enregistrement des métriques avec MLflow
    with mlflow.start_run():
        rmse_test, r2_test = baseline_model.get_score()
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_test", r2_test)
    mlflow.end_run()

    # Modèle XGBoost
    logging.info('Début de l\'entraînement du modèle XGBoost.')
    xgboost_model = XGBoostModel()
    xgboost_pipeline = XGBoostPipeline(xgboost_model)
    xgboost_pipeline.train_fit(train_features, test_features, train_target, test_target)
    logging.info('Fin de l\'entraînement du modèle XGBoost.')
    # Évaluation du modèle et enregistrement des métriques
    rmse_test, r2_test = xgboost_model.get_score()
    logging.info('Métriques de test du modèle XGBoost:')
    logging.info('RMSE: %f', rmse_test)
    logging.info('R2: %f', r2_test)
    # Enregistrement des métriques avec MLflow
    best_params = xgboost_model.get_best_params()
    with mlflow.start_run():
        rmse_test, r2_test = xgboost_model.get_score()
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_test", r2_test)
        mlflow.log_params(best_params)
    mlflow.end_run()

if __name__ == "__main__":
    main()
