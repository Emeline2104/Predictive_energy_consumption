"""
Ce fichier contient le script principal du projet.

Il utilise les modules préalablement définis pour nettoyer les données, 
effectuer un ingénierie des fonctionnalités, 
et entraîner des modèles d'apprentissage automatique.
"""

# Importation des modules nécessaires
import pandas as pd
import logging
from scr.preprocessing.data_cleaning import DataCleaningPipeline, DataCleaner
from scr.preprocessing.feature_engineering import DataFeatureEngineering, FeatureEngineeringPipeline
from scr.models.xgboost_model import XGBoostModel, XGBoostPipeline
from scr.models.baseline_model import RLModel, RLModelPipeline
from scr.config import data_file_path

def main():
    """
    Fonction principale pour l'exécution des étapes de nettoyage, d'ingénierie des features
    et d'entraînement des modèles.

    Returns:
        None
    """
    # Chargement des données
    data_conso = pd.read_csv(data_file_path) # A mettre en maj -> var globale

    # Nettoyage des données
    cleaning_pipeline = DataCleaningPipeline(DataCleaner)
    train_features, test_features, train_target, test_target = cleaning_pipeline.transform(
        data_conso,
        )

    # Exécute le pipeline de feature engineering sur les données nettoyées
    feature_engineering_pipeline = FeatureEngineeringPipeline(DataFeatureEngineering)
    train_features, test_features, train_target, test_target = feature_engineering_pipeline.transform(
        train_features,
        test_features,
        train_target,
        test_target,
        )

    # Modèle de base (RL)
    baseline_model = RLModel()
    baseline_pipeline = RLModelPipeline(baseline_model)
    baseline_pipeline.run_pipeline(train_features, test_features, train_target, test_target)

    # Modèle XGBoost
    xgboost_model = XGBoostModel()
    xgboost_pipeline =   XGBoostPipeline(xgboost_model)
    xgboost_pipeline.run_pipeline(train_features, test_features, train_target, test_target)

if __name__ == "__main__":
    main()
