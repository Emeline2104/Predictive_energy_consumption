"""
Ce fichier contient le script principal du projet.

Il utilise les modules préalablement définis pour nettoyer les données, 
effectuer un ingénierie des fonctionnalités, 
et entraîner des modèles d'apprentissage automatique.
"""

# Importation des modules nécessaires
import pandas as pd
from preprocessing.data_cleaning import Cleaning, DataCleaningPipeline
from preprocessing.feature_engineering import FeatureEngineering, FeatureEngineeringPipeline
from models.xgboost_model import XGBoostModel, XGBoostPipeline
from models.baseline_model import RLModel, RLModelPipeline
from config import data_file_path

def main(data): 
    """
    Fonction principale pour l'exécution des étapes de nettoyage, d'ingénierie des features
    et d'entraînement des modèles.

    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données à traiter.

    Returns:
        None
    """
    # Nettoyage des données
    data_cleaning_pipeline = DataCleaningPipeline(steps=[
        Cleaning.extract_building_types,
        Cleaning.drop_outliers,
        Cleaning.remove_missing_data,
        Cleaning.apply_usage_correspondence,
    ])

    # Exécute le pipeline de nettoyage sur les données
    train_features, test_features, train_target, test_target = data_cleaning_pipeline.run_pipeline(data)

    # Feature Engineering
    liste_utilisation_target = ['Categorie1_Encoded', 'Categorie2_Encoded', 'Categorie3_Encoded']
    group_columns_utilisation = [
        'TypeUtilisationPrincipale',
        'TypeUtilisationSecondaire',
        'TypeUtilisationTertiaire',
    ]

    features_engineering_pipeline = FeatureEngineeringPipeline(steps=[
        FeatureEngineering.process_physical_data,
        FeatureEngineering.add_age_feature,
        lambda train_features, test_features, train_target, test_target: FeatureEngineering.encode_target(
            train_features, test_features, train_target, test_target,
            group_columns=group_columns_utilisation,
            new_column_names=liste_utilisation_target,
        ),
        FeatureEngineering.preprocess_neighborhood_data,
        lambda train_features, test_features, train_target, test_target: FeatureEngineering.encode_target(
            train_features, test_features, train_target, test_target,
            group_columns=['Neighborhood'],
            new_column_names=['Neighborhood_Encoded'],
        ),
        FeatureEngineering.clean_numerical_columns,
        FeatureEngineering.clean_data,
    ])

    # Exécute le pipeline de feature engineering sur les données nettoyées
    train_features, test_features, train_target, test_target = features_engineering_pipeline.run_pipeline(train_features, test_features, train_target, test_target)

    # Modèle de base (RL)
    print("Réalisation de la regression Baseline (RL)")
    baseline_model = RLModel()
    baseline_pipeline = RLModelPipeline(baseline_model)
    baseline_pipeline.run_pipeline(train_features, test_features, train_target, test_target)
    print("####################")

    # Modèle XGBoost
    print("Réalisation de la regression XGBoost")
    xgboost_model = XGBoostModel()
    xgboost_pipeline = XGBoostPipeline(xgboost_model)  
    # plus dissiqué pour faire apparaître les méthodes ?
    xgboost_pipeline.run_pipeline(train_features, test_features, train_target, test_target)

if __name__ == "__main__":
    data_conso = pd.read_csv(data_file_path)
    main(data_conso)

