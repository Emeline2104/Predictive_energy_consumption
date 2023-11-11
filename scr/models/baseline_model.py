"""
Module : baseline_model

Ce module définit la classe RLModel pour l'entraînement, la validation croisée
et l'évaluation d'un modèle de régression linéaire. Il définit également la
classe RLModelPipeline pour encapsuler l'ensemble du workflow.

Classes :
- RLModel : Classe pour l'entraînement, la validation croisée et l'évaluation 
des modèles de régression linéaire.
- RLModelPipeline : Classe pour encapsuler l'ensemble du workflow d'entraînement,
de validation et d'évaluation en utilisant RLModel.

"""
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

class RLModel:
    """
    Classe représentant un modèle de régession lineaire.

    Attributs :
    - model : Instance du modèle LinearRegression.
    - rmse_val : Score RMSE moyen de la validation croisée.
    - r2_val : Score R2 moyen de la validation croisée.
    - rmse_test : Score RMSE sur les données de test.
    - r2_test : Score R2 sur les données de test.

    Méthodes :
    - train : Entraîne le modèle.
    - cross_validate : Effectue une validation croisée pour évaluer les performances du modèle.
    - predict_and_evaluate : Prédit et évalue le modèle sur les données de test.
    - get_best_model : Retourne le meilleur modèle entraîné.
    """
    def __init__(self):
        """
        Initialise un objet RLModel.

        Attributs :
        - model : Modèle de régression lineaire.
        - rmse_val : Erreur quadratique moyenne (RMSE) lors de la validation croisée.
        - r2_val : Score R carré (R2) lors de la validation croisée.
        - rmse_test : RMSE sur les données de test.
        - r2_test : Score R2 sur les données de test.
        """
        self.model = LinearRegression()
        self.rmse_val = None
        self.r2_val = None
        self.rmse_test = None
        self.r2_test = None

    def train(self, x_train, y_train):
        """
        Entraîne le modèle de régression linéaire.

        :param x_train: Données d'entraînement.
        :param y_train: Variable cible pour les données d'entraînement.
        :param param_grid: Grille d'hyperparamètres pour la recherche en grille.
                            Si None, une grille par défaut est utilisée.
        """
        self.model.fit(x_train, y_train)

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
        print(f"R2 Score de test: {self.r2_test:.2f}")

    def get_best_model(self):
        """
        Obtient le modèle de régression XGBoost entraîné.
        """
        return self.model

class RLModelPipeline:
    """
    Classe représentant un pipeline regression linéaire complet.

    Attributs :
    - model_trainer : Instance de la classe RLModel utilisée pour l'entraînement 
                        et l'évaluation du modèle.

    Méthodes :
    - run_pipeline : Exécute le pipeline complet, comprenant l'entraînement, la validation croisée, 
                la prédiction et l'évaluation du modèle.
    """
    def __init__(self, model_trainer):
        """
        Initialise un objet RLModelPipeline.

        :param model_trainer: Objet RLModel pour l'entraînement et l'évaluation.
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
