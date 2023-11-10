"""
Module : xgboost_model

Ce module définit la classe XGBoostModel pour l'entraînement, la validation croisée
et l'évaluation d'un modèle de régression XGBoost. Il définit également la
classe XGBoostPipeline pour encapsuler l'ensemble du workflow.

Classes :
- XGBoostModel : Classe pour l'entraînement, la validation croisée et l'évaluation 
des modèles de régression XGBoost.
- XGBoostPipeline : Classe pour encapsuler l'ensemble du workflow d'entraînement,
de validation et d'évaluation en utilisant XGBoostModel.

"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARAM_GRID_DEFAUT_XGBOOST = {
        'n_estimators': [100, 200, 300, 500],  # Nombre d'estimateurs
        'max_depth': [3, 5, 7],  # Profondeur maximale de chaque arbre
        'learning_rate': [0.01, 0.1, 0.2],  # Taux d'apprentissage
        'subsample': [0.8, 1.0],  # Sous-échantillonnage des observations
        'colsample_bytree': [0.8, 1.0],  # Sous-échantillonnage des colonnes
        'gamma': [0, 1, 5],  # Valeur de réduction minimale requise pour effectuer une division
    }

class XGBoostModel:
    """
    Classe représentant un modèle XGBoost pour la régression.

    Attributs :
    - model : Instance du modèle XGBoost.
    - best_params : Meilleurs paramètres trouvés par la recherche sur grille.
    - rmse_val : Score RMSE moyen de la validation croisée.
    - r2_val : Score R2 moyen de la validation croisée.
    - rmse_test : Score RMSE sur les données de test.
    - r2_test : Score R2 sur les données de test.

    Méthodes :
    - train : Entraîne le modèle en utilisant la recherche sur grille pour trouver 
                les meilleurs paramètres.
    - cross_validate : Effectue une validation croisée pour évaluer les performances du modèle.
    - predict_and_evaluate : Prédit et évalue le modèle sur les données de test.
    - get_feature_importance : Calcule et affiche l'importance des caractéristiques.
    - get_best_model : Retourne le meilleur modèle entraîné.
    - get_best_params : Retourne les meilleurs paramètres trouvés par la recherche sur grille.
    """
    def __init__(self):
        """
        Initialise un objet XGBoostModel.

        Attributs :
        - model : Modèle de régression XGBoost.
        - best_params : Meilleurs hyperparamètres obtenus lors de la recherche en grille.
        - rmse_val : Erreur quadratique moyenne (RMSE) lors de la validation croisée.
        - r2_val : Score R carré (R2) lors de la validation croisée.
        - rmse_test : RMSE sur les données de test.
        - r2_test : Score R2 sur les données de test.
        """
        self.model = XGBRegressor()
        self.best_params = None
        self.rmse_val = None
        self.r2_val = None
        self.rmse_test = None
        self.r2_test = None

    def train(self, x_train, y_train, param_grid=None):
        """
        Entraîne le modèle de régression XGBoost.

        :param x_train: Données d'entraînement.
        :param y_train: Variable cible pour les données d'entraînement.
        :param param_grid: Grille d'hyperparamètres pour la recherche en grille.
                            Si None, une grille par défaut est utilisée.
        """
        if param_grid is None:
            param_grid = PARAM_GRID_DEFAUT_XGBOOST

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5)
        grid_search.fit(x_train, y_train.ravel())
        best_params = grid_search.best_params_

        self.model = XGBRegressor(**best_params)
        self.model.fit(x_train, y_train)
        self.best_params = best_params

    def cross_validate(self, x_train, y_train):
        """
        Effectue une validation croisée.

        :param x_train: Données d'entraînement.
        :param y_train: Variable cible pour les données d'entraînement.
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
        for train_index, test_index in stratified_kfold.split(x_train, y_bin_labels):
            x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_fold = pd.DataFrame(y_train).iloc[train_index]
            y_test_fold = pd.DataFrame(y_train).iloc[test_index]

            self.model.fit(x_train_fold, y_train_fold)
            y_pred_fold = self.model.predict(x_test_fold)

            # Evaluation des performances du modèle sur ce pli
            rmse_fold = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
            r2_fold = r2_score(y_test_fold, y_pred_fold)

            # Ajout des scores à la liste respective
            rmse_scores.append(rmse_fold)
            r2_scores.append(r2_fold)

        self.rmse_val = np.mean(rmse_scores)
        self.r2_val = np.mean(r2_scores)
        print(f"Moyenne de la validation croisée RMSE: {self.rmse_val:.2f}")
        print(f"Moyenne de la validation croisée R2 Score: {self.r2_val:.2f}")



    def predict_and_evaluate(self, x_test, y_test):
        """
        Prédit et évalue sur les données de test.

        :param x_test: Données de test.
        :param y_test: Variable cible pour les données de test.
        """
        # Score sur les données de test
        y_pred = self.model.predict(x_test)
        self.rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE de test: {self.rmse_test:.2f}")
        self.r2_test = r2_score(y_test, y_pred)
        print(f"R2 Score de test: {self.rmse_test:.2f}")

    def get_feature_importance(self, x_train):
        """
        Calcule l'importance des caractéristiques pour le meilleur modèle.

        Affiche un graphique des importances des caractéristiques.

        :param x_train: Données d'entraînement standardisées.
        """
        importances = self.model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        feature_importances = feature_importances.iloc[0:25, :]

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.show()

    def get_best_model(self):
        """
        Obtient le modèle de régression XGBoost entraîné.
        """
        return self.model

    def get_best_params(self):
        """
        Obtient les meilleurs hyperparamètres obtenus lors de la recherche en grille.
        """
        return self.best_params

class XGBoostPipeline:
    """
    Classe représentant un pipeline XGBoost complet.

    Attributs :
    - model_trainer : Instance de la classe XGBoostModel utilisée pour l'entraînement 
                        et l'évaluation du modèle.

    Méthodes :
    - run_pipeline : Exécute le pipeline complet, comprenant l'entraînement, la validation croisée, 
                la prédiction, l'évaluation et l'affichage de l'importance des caractéristiques.
    """
    def __init__(self, model_trainer):
        """
        Initialise un objet XGBoostPipeline.

        :param model_trainer: Objet XGBoostModel pour l'entraînement et l'évaluation.
        """
        self.model_trainer = model_trainer

    def run_pipeline(self, x_train, x_test, y_train, y_test):
        """
        Exécute le pipeline complet.

        :param x_train: Données d'entraînement.
        :param x_test: Données de test.
        :param y_train: Variable cible des données d'entraînement.
        :param y_test: Variable cible des données de test.
        """
        # Étape 1: Entraînement du modèle
        self.model_trainer.train(x_train, y_train)

        # Étape 2: Validation croisée
        self.model_trainer.cross_validate(x_train, y_train)

        # Étape 3: Prédiction et évaluation sur les données de test
        self.model_trainer.predict_and_evaluate(x_test, y_test)

        # Étape 4: Affichage de l'importance des caractéristiques
        self.model_trainer.get_feature_importance(x_train)
    
