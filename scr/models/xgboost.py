from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scr.preprocessing.data_cleaning import Cleaning, DataCleaningPipeline
from preprocessing.feature_engineering import features_engineering_pipeline

PARAM_GRID_DEFAUT_XGBOOST = {
        'n_estimators': [100, 200, 300, 500],  # Nombre d'estimateurs
        'max_depth': [3, 5, 7],  # Profondeur maximale de chaque arbre
        'learning_rate': [0.01, 0.1, 0.2],  # Taux d'apprentissage
        'subsample': [0.8, 1.0],  # Sous-échantillonnage des observations
        'colsample_bytree': [0.8, 1.0],  # Sous-échantillonnage des colonnes
        'gamma': [0, 1, 5],  # Valeur de réduction minimale requise pour effectuer une division
    }       

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor()
        self.best_params = None
        self.rmse_val = None
        self.r2_val = None
        self.rmse_test = None
        self.r2_test = None

    def train(self, X_train, y_train, param_grid=None):
        if param_grid is None: 
            param_grid = PARAM_GRID_DEFAUT_XGBOOST
            
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train.ravel())
        best_params = grid_search.best_params_
        
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X_train, y_train)
        self.best_params = best_params
    
    def cross_validate(self, X_train, y_train):
        """
        Effectue une validation croisée.

        :param X_train_stand: Données d'entraînement normalisées.
        :param y_train: Variable cible des données d'entraînement.
        :param best_model: Meilleur modèle obtenu.
        :return: Scores RMSE et R2 de la validation croisée.
        """

        # Discrétisation de la variable cible en bins pour stratification de la cross-validation
        num_bins = 5
        y_bins = np.linspace(y_train.min(), y_train.max(), num_bins + 1)
        # Assignation des bins aux observations
        y_bin_labels = np.digitize(y_train, y_bins)
        # Listes pour stocker les scores de chaque fold
        rmse_scores = []
        r2_scores = []

        # Stratification des plis en utilisant StratifiedKFold
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        # Boucle de validation croisée
        for train_index, test_index in stratified_kfold.split(X_train, y_bin_labels):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = pd.DataFrame(y_train).iloc[train_index], pd.DataFrame(y_train).iloc[test_index]

            self.model.fit(X_train_fold, y_train_fold)
            y_pred_fold = self.model.predict(X_test_fold)
        
            # Evaluation des performances du modèle sur ce pli
            rmse_fold = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
            r2_fold = r2_score(y_test_fold, y_pred_fold)
        
            print("RMSE fold: {:.2f}".format(rmse_fold))
            print("R2 Score fold:", r2_fold)
            # Ajout des scores à la liste respective
            rmse_scores.append(rmse_fold)
            r2_scores.append(r2_fold)

        self.rmse_val = np.mean(rmse_scores)
        self.r2_val = np.mean(r2_scores)
        print("Moyenne de la validation croisée RMSE: {:.2f}".format(self.rmse_val ))
        print("Moyenne de la validation croisée R2 Score: {:.2f}".format(self.r2_val))

    
    def predict_and_evaluate(self, X_test, y_test):
        """
        Prédit et évalue sur les données de test.

        :param best_model: Meilleur modèle obtenu.
        :param X_test_stand: Données de test normalisées.
        :param y_test: Variable cible des données de test.
        :return: Scores RMSE et R2 sur les données de test.
        """
        # Score sur les données de test
        y_pred = self.model.predict(X_test)
        self.rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE de test : {:.2f}".format(self.rmse_test))
        self.r2_test = r2_score(y_test, y_pred)
        print("R2 Score de test:", self.r2_test)
    
    def get_feature_importance(self, X_train):
        """
        Calcule l'importance des caractéristiques pour le meilleur modèle.
        
        Affiche un graphique des importances des caractéristiques.

        :param best_model: Le modèle pour lequel l'importance des caractéristiques est calculée.
        :param X_train_stand: Les données d'entraînement standardisées.
        :param y_pred: Les prédictions.

        :return: Affiche le graphique des importances des caractéristiques.
        """
        importances = self.model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        feature_importances = feature_importances.iloc[0:25, :]

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.show()

    def get_best_model(self):
        return self.model
    
    def get_best_params(self):
        return self.best_params

class XGBoostPipeline:
    def __init__(self, data_preprocessor, model_trainer):
        self.data_preprocessor = data_preprocessor
        self.model_trainer = model_trainer
    
    def run_pipeline(self, X_train, X_test, y_train, y_test):
        """
        Exécute le pipeline complet.

        :param X_train: Données d'entraînement.
        :param X_test: Données de test.
        :param y_train: Variable cible des données d'entraînement.
        :param y_test: Variable cible des données de test.
        """
        # Étape 1: Prétraitement des données
        X_train, X_test, y_train, y_test = self.data_preprocessor.clean(X_train, X_test, y_train, y_test)
        
        # Étape 2: Entraînement du modèle
        self.model_trainer.train(X_train, y_train)
        
        # Étape 3: Validation croisée
        self.model_trainer.cross_validate(X_train, y_train)
        
        # Étape 4: Prédiction et évaluation sur les données de test
        self.model_trainer.predict_and_evaluate(X_test, y_test)
        
        # Étape 5: Affichage de l'importance des caractéristiques
        self.model_trainer.get_feature_importance(X_train)


# Exemple d'utilisation du pipeline
data_preprocessor = features_engineering_pipeline

model_trainer = XGBoostModel() # utile ? ne pas faire dans le main ? ajouter cleaner ? 
xgboost_pipeline = XGBoostPipeline(data_preprocessor, model_trainer)

data_cleaning_pipeline = DataCleaningPipeline(Cleaning(data))
X_train, X_test, y_train, y_test = data_cleaning_pipeline.clean()

# Utilisation du pipeline avec vos données
xgboost_pipeline.run_pipeline(X_train, X_test, y_train, y_test)










