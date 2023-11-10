"""
Fichier : feature_engineering.py

Ce fichier regroupe des classes pour le nettoyage de données.

Classes :
- FeatureEngineering : Regroupe des méthodes spécifiques aux feature engineering.
Méthodes : encode_target, process_physical_data, add_age_feature, 
preprocess_neighborhood_data, clean_numerical_columns, clean_data
- FeatureEngineering : Pipeline de nettoyage de données avec une étape de nettoyage 
de type FeatureEngineering.
Méthode : clean.

Utilisation :
Charger les données néttoyer (x_train, y_train, x_test, y_test).
Créer une instance de FeatureEngineering avec ces données.
Créer une instance de FeatureEngineeringPipeline avec l'instance de FeatureEngineering.
Exécuter le feature engineering en appelant la méthode clean de la FeatureEngineeringPipeline.
Afficher les données nettoyées.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    """
    Regroupe les méthodes de feature engineering spécifiques, comme le target encoding,
    le preproccess des données physiques, la gestion de l'âge des bâtiments, 
    le preproccess des données géographiques, le nettoyage des données non sélectionnées. 
    Chaque méthode effectue une tâche spécifique de feature engineering.
    """
    @staticmethod
    def encode_target(x_train, x_test, y_train, y_test, group_columns, new_column_names):
        """
        Encode plusieurs variables catégorielles en utilisant la moyenne de la variable cible 
        par catégorie.

        :param x_train: DataFrame d'entraînement.
        :param x_test: DataFrame de test.
        :param y_train: Variable cible des données d'entraînement.
        :param y_test: Variable cible des données de test.
        :param group_columns: Liste des noms des colonnes catégorielles à encoder.
        :param new_column_names: Liste des noms de nouvelles colonnes qui stockeront 
        les moyennes encodées.

        :return: DataFrame avec les nouvelles colonnes encodées.
        """
        indic_conso = ['SiteEnergyUse(kBtu)']

        for i, column in enumerate(group_columns):
            # Calcul des moyennes de la variable cible par catégorie
            target_mean = y_train.groupby(x_train[column])[indic_conso[0]].mean()

            # Encodage de la variable catégorielle en utilisant les moyennes
            x_train[new_column_names[i]] = x_train[column].map(target_mean)
            x_test[new_column_names[i]] = x_test[column].map(target_mean)

            # Remplacement des valeurs manquantes par 0 (ou une autre valeur au choix)
            x_train[new_column_names[i]].fillna(0, inplace=True)
            x_test[new_column_names[i]].fillna(0, inplace=True)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def process_physical_data(x_train, x_test, y_train, y_test):
        """
        Traite les données physiques en calculant et transformant les variables d'intérêt.

        Args:
            x_train (pd.DataFrame): DataFrame d'entraînement contenant les données physiques.
            x_test (pd.DataFrame): DataFrame de test contenant les données physiques.

        Returns:
            x_train (pd.DataFrame): DataFrame d'entraînement avec les variables physiques 
            transformées.
            x_test (pd.DataFrame): DataFrame de test avec les variables physiques transformées.
        """

        # Copie explicite du DataFrame pour éviter les avertissements de SettingWithCopyWarning
        x_train = x_train.copy()
        x_test = x_test.copy()

        # Calcul de la variable 'Densite_etage' en prenant en compte la densité d'étage par bâtiment
        x_train['Densite_etage'] = x_train['NumberofFloors'] / (x_train['NumberofBuildings'] + 1)
        x_test['Densite_etage'] = x_test['NumberofFloors'] / (x_test['NumberofBuildings'] + 1)

        # Passage au log des données asymétriques
        columns_to_log = ['PropertyGFABuilding(s)', 'PropertyGFAParking', 'Densite_etage']
        for colonne in columns_to_log:
            x_train[colonne] =x_train[colonne].apply(lambda x: np.log(x) if x != 0 else 0)
            x_test[colonne] = x_test[colonne].apply(lambda x: np.log(x) if x != 0 else 0)
        print("Columns in x_train:", x_train.columns)
        print("Columns in x_test:", x_test.columns)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def add_age_feature(x_train, x_test, y_train, y_test):
        """
        Ajoute la variable 'Age' aux données en soustrayant l'année de construction 
        à l'année actuelle.

        Args:
            x_train (pd.DataFrame): DataFrame d'entraînement contenant les données.
            x_test (pd.DataFrame): DataFrame de test contenant les données.
            year_column (str): Nom de la colonne contenant les années de construction.

        Returns:
            x_train (pd.DataFrame): DataFrame d'entraînement avec la variable 'Age' ajoutée.
            x_test (pd.DataFrame): DataFrame de test avec la variable 'Age' ajoutée.
        """
        year_column='YearBuilt'
        current_year = 2023
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train['Age'] = current_year - x_train[year_column]
        x_test['Age'] = current_year - x_test[year_column]
        print("Columns in x_train:", x_train.columns)
        print("Columns in x_test:", x_test.columns)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def preprocess_neighborhood_data(x_train, x_test, y_train, y_test):
        """
        Met en majuscule les données de quartiers pour les homogénéiser et corrige
        une erreur de saisie.

        Args:
            x_train (pandas.DataFrame): DataFrame d'entraînement avec la colonne 'Neighborhood'.
            x_test (pandas.DataFrame): DataFrame de test avec la colonne 'Neighborhood'.

        Returns:
            x_train (pd.DataFrame): DataFrame d'entraînement.
            x_test (pd.DataFrame): DataFrame de test.
        """
        # Mettre en majuscule les données de quartiers pour homogénéiser
        x_train['Neighborhood'] = x_train['Neighborhood'].str.capitalize()
        x_test['Neighborhood'] = x_test['Neighborhood'].str.capitalize()

        # Correction d'une erreur de saisie des quartiers
        x_train.loc[x_train['Neighborhood'] == 'Delridge neighborhoods','Neighborhood'] = 'Delridge'
        x_test.loc[x_test['Neighborhood'] == 'Delridge neighborhoods','Neighborhood'] = 'Delridge'

        return x_train, x_test, y_train, y_test

    @staticmethod
    def clean_numerical_columns(x_train, x_test, y_train, y_test):
        """
        Nettoie et supprime des colonnes spécifiques du jeu de données.

        Args:
            x_train (pd.DataFrame): Données d'entraînement.
            x_test (pd.DataFrame): Données de test.
            colonnes_a_sup (list): Liste des colonnes à supprimer.

        Returns:
            x_train (pd.DataFrame): Données d'entraînement nettoyées.
            x_test (pd.DataFrame): Données de test nettoyées.
        """
        colonnes_a_sup = ['DataYear',
                    'ZipCode',
                    'CouncilDistrictCode',
                    'Latitude',
                    'Longitude', 
                    'YearBuilt',
                    'NumberofBuildings',
                    'NumberofFloors', 
                    'PropertyGFATotal',
                    'LargestPropertyUseTypeGFA',
                    'SecondLargestPropertyUseTypeGFA',
                    'ThirdLargestPropertyUseTypeGFA', 
                    'ENERGYSTARScore',
                    'SiteEUIWN(kBtu/sf)', 
                    'SourceEUI(kBtu/sf)',
                    'SourceEUIWN(kBtu/sf)',
                    'SiteEnergyUseWN(kBtu)',
                    'SteamUse(kBtu)',
                    'Electricity(kWh)',
                    'Electricity(kBtu)',
                    'NaturalGas(therms)',
                    'NaturalGas(kBtu)',
                    'Comments',
                    'TotalGHGEmissions',
                    'GHGEmissionsIntensity',
                    'SiteEUI(kBtu/sf)',
                    'OSEBuildingID',
                    ]

        x_train = x_train.select_dtypes(exclude=['object'])
        x_test = x_test.select_dtypes(exclude=['object'])
        x_train.drop(columns=colonnes_a_sup, inplace=True)
        x_test.drop(columns=colonnes_a_sup, errors='ignore', inplace=True)
        print("Columns in x_train:", x_train.columns)
        print("Columns in x_test:", x_test.columns)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def clean_data(x_train, x_test, y_train, y_test):
        """
        Nettoie les données d'entraînement et de test.

        :param x_train: Données d'entraînement.
        :param x_test: Données de test.
        :param y_train: Variable cible des données d'entraînement.
        :param indic_conso: Indicateurs de consommation.
        :return: Données d'entraînement et de test nettoyées.
        """

        # Suppression des lignes avec des valeurs manquantes
        x_train = x_train.drop(x_train.loc[x_train['DefaultData'] == 'True'].index)
        x_test = x_test.drop(x_test.loc[x_test['DefaultData'] == 'True'].index)

        # Suppression de la colonne DefaultData
        x_train = x_train.drop(['DefaultData'], axis=1)
        x_test = x_test.drop(['DefaultData'], axis=1)

        # Remplacement des NaN par 0
        x_train = x_train.fillna(0)
        x_test = x_test.fillna(0)

        #standardisation
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        # conversion y_train en un tableau 1D
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        return x_train, x_test, y_train, y_test

class FeatureEngineeringPipeline:
    """
    Classe représentant un pipeline de feature engineering.
    """
    def __init__(self, steps):
        self.steps = steps

    def run_pipeline(self, x_train, x_test, y_train, y_test):
        """
        Applique les étapes de nettoyage spécifiées dans la pipeline.

        Args:
            x_train (pd.DataFrame): Les données d'entraînement.
            x_test (pd.DataFrame): Les données de test.
            y_train (pd.Series): La variable cible des données d'entraînement.
            y_test (pd.Series): La variable cible des données de test.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Les données nettoyées 
            après avoir appliqué chaque étape.
        """
        for step in self.steps:
            x_train, x_test, y_train, y_test = step(x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test

