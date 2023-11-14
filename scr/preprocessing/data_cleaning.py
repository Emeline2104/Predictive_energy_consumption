"""
Fichier : data_cleaning.py

Ce fichier regroupe des classes pour le nettoyage de données.


Classes :
- DataCleaner : Regroupe des méthodes spécifiques de nettoyage de données.
Méthodes : extract_building_types, drop_outliers, remove_missing_data, apply_usage_correspondence,
split_data
- DataCleaningPipeline : Pipeline de nettoyage de données avec une étape de nettoyage 
de type DataCleaner.
Méthode : transform.

Utilisation :
Charger les données à nettoyer depuis un fichier CSV.
Créer une instance de DataCleaner avec ces données.
Créer une instance de DataCleaningPipeline avec l'instance de DataCleaner.
Exécuter le nettoyage en appelant la méthode transform de la classe DataCleaningPipeline.
Afficher les données nettoyées.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

class DataCleaner:
    """
    Regroupe les méthodes de nettoyage spécifiques, comme l'extraction des types de bâtiments,
    la suppression des valeurs aberrantes, la suppression des données manquantes, 
    et l'application de la correspondance des usages.
    Chaque méthode effectue une tâche spécifique de nettoyage correspondant à un transformer.
    """

    @staticmethod
    def extract_building_types(data):
        """
        Extrait les données des types de bâtiments spécifiques pour garder les bâtiments 
        non résidentiels.

        Returns:
        pandas.DataFrame: Le DataFrame filtré selon les critères spécifiés.
        """
        # Extraction des bâtiments non résidentiels
        building_types_to_keep = ['NonResidential', 'Nonresidential WA', 'Nonresidential COS']
        data = data[data['BuildingType'].isin(building_types_to_keep)]

        # Suppression de données spécifiques
        data_to_remove = [
            'Other - Lodging/Residential',
            'Residence Hall/Dormitory',
            'Residential Care Facility',
            'Multifamily Housing',
        ]
        mask = data['LargestPropertyUseType'].isin(data_to_remove)
        data = data[~mask]

        return data

    @staticmethod
    def drop_outliers(data):
        """
        Supprime les valeurs aberrantes du DataFrame selon la colonne 'Outlier' 
        et la 'ComplianceStatus'.

        Returns:
        pandas.DataFrame: Le DataFrame sans les valeurs aberrantes.
        """
        data = data[
            (~data['Outlier'].isin(['High outlier', 'Low outlier']))
            & (data['ComplianceStatus'] == 'Compliant')
        ]

        return data

    @staticmethod
    def remove_missing_data(data):
        """
        Supprime les lignes contenant des données manquantes dans un DataFrame.

        Returns:
        pd.DataFrame: Le DataFrame nettoyé.
        """
        data = data.dropna(
            axis=0,
            how='any',
            subset=['LargestPropertyUseType', 'SiteEnergyUse(kBtu)']
        )
        return data

    @staticmethod
    def apply_usage_correspondence(data):
        """
        Applique une table de correspondance des usages à un DataFrame pour mapper les valeurs.

        Returns:
        pd.DataFrame: Le DataFrame avec les colonnes d'usage mappées selon la correspondance.
        """
        # Mapping des usages selon une table de correspondance
        correspondence_usage = {
            # faire fichier JSON ?
            'Data Center': 'Data Center',
            'Urgent Care/Clinic/Other Outpatient': 'Médical',
            'Laboratory': 'Médical',
            'Hospital (General Medical & Surgical)': 'Médical',
            'Supermarket/Grocery Store': 'Magasin',
            'Restaurant': 'Restaurant',
            'Other/Specialty Hospital': 'Médical',
            'Museum': 'Divertissement',
            'Other - Restaurant/Bar': 'Restaurant',
            'Other - Recreation': 'Divertissement',
            'Police Station': 'Service public',
            'Parking': 'Parking',
            'Lifestyle Center': 'Sport',
            'Senior Care Community': 'Service',
            'Other - Education': 'Education',
            'Personal Services (Health/Beauty, Dry Cleaning...': 'Service à la personne',
            'Manufacturing/Industrial Plant': 'Usine industrielle',
            'Fitness Center/Health Club/Gym': 'Sport',
            'Wholesale Club/Supercenter': 'Magasin',
            'Medical Office': 'Médical',
            'Other': 'Autres',
            'Social/Meeting Hall': 'Divertissement',
            'Strip Mall': 'Supermarché',
            'Other - Entertainment/Public Assembly': 'Divertissement',
            'Other - Public Services': 'Service public',
            'Courthouse': 'Service public',
            'Movie Theater': 'Divertissement',
            'Hotel': 'Hotel',
            'Fire Station': 'Service public',
            'Refrigerated Warehouse': 'Stockage',
            'College/University': 'Education',
            'Financial Office': 'Bureaux',
            'Library': 'Education',
            'Bank Branch': 'Service',
            'Adult Education': 'Education',
            'Retail Store': 'Magasin',
            'Prison/Incarceration': 'Service public',
            'Office': 'Bureaux',
            'Other - Services': 'Service',
            'Performing Arts': 'Divertissement',
            'Repair Services (Vehicle, Shoe, Locksmith, etc)': 'Service',
            'K-12 School': 'Education',
            'Pre-school/Daycare': 'Education',
            'Automobile Dealership': 'Autres',
            'Other - Utility': 'Autres',
            'Non-Refrigerated Warehouse': 'Stockage',
            'Distribution Center': 'Magasin',
            'Worship Facility': 'Lieu de culte',
            'Self-Storage Facility': 'Stockage',
            'Other - Mall': 'Supermarché',
            'Food Service': 'Restaurant',
            'Personal Services (Health/Beauty, Dry Cleaning, etc)': 'Service',
        }
        usage_columns = [
            'LargestPropertyUseType',
            'SecondLargestPropertyUseType',
            'ThirdLargestPropertyUseType',
        ]
        final_usages = [
            'TypeUtilisationPrincipale',
            'TypeUtilisationSecondaire',
            'TypeUtilisationTertiaire',
        ]

        for i, col in enumerate(usage_columns):
            data[final_usages[i]] = data[col].map(correspondence_usage)

        return data

    @staticmethod
    def split_data(data):
        """
        Prépare les données en séparant les features et la target, effectue la 
        stratification et renvoie les ensembles d'entraînement et de test.

        Returns:
        x_train (pd.DataFrame): Ensemble d'entraînement des features.
        x_test (pd.DataFrame): Ensemble de test des features.
        y_train (pd.DataFrame): Ensemble d'entraînement de la target.
        y_test (pd.DataFrame): Ensemble de test de la target.
        """
        indic_conso = ['SiteEnergyUse(kBtu)']

        # Crée une table de features en supprimant les colonnes d'indicateurs de consommation
        features = data.drop(columns=indic_conso)

        # Crée une table de target en incluant uniquement les colonnes d'indicateurs de consommation
        target = data[indic_conso]

        # Convertit la colonne 'TypeUtilisationPrincipale' en chaîne de caractères
        features['TypeUtilisationPrincipale'] = features['TypeUtilisationPrincipale'].astype(str)

        # Stratification sur la variable catégorielle pour obtenir x_train et x_test
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=None,
            stratify=features['TypeUtilisationPrincipale'],
        )

        print("Dimensions de l'ensemble d'entraînement :", x_train.shape, y_train.shape)
        print("Dimensions de l'ensemble de test :", x_test.shape, y_test.shape)

        # Conversion des arrays NumPy en DataFrames
        x_train = pd.DataFrame(x_train, columns=features.columns)
        x_test = pd.DataFrame(x_test, columns=features.columns)
        y_train = pd.DataFrame(y_train, columns=target.columns)
        y_test = pd.DataFrame(y_test, columns=target.columns)

        return x_train, x_test, y_train, y_test

class DataCleaningPipeline:
    """
    Classe représentant un pipeline de nettoyage de données.

    Attributes:
    transformer (DataCleaner): Le transformateur utilisé pour nettoyer les données.
    x_train (pd.DataFrame): Ensemble d'entraînement des features.
    x_test (pd.DataFrame): Ensemble de test des features.
    y_train (pd.DataFrame): Ensemble d'entraînement de la target.
    y_test (pd.DataFrame): Ensemble de test de la target.
    """
    def __init__(self, transformer):
        self.transformer = transformer
        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.y_train = None

    def transform(self, data):
        """
        Transforme les données en appliquant les étapes de nettoyage spécifiées 
        par le transformateur.

        Args:
        data (pd.DataFrame): Le DataFrame contenant les données à nettoyer.

        Returns:
        x_train (pd.DataFrame): Ensemble d'entraînement des features.
        x_test (pd.DataFrame): Ensemble de test des features.
        y_train (pd.DataFrame): Ensemble d'entraînement de la target.
        y_test (pd.DataFrame): Ensemble de test de la target.
        """
        data = self.transformer.extract_building_types(data)
        data = self.transformer.drop_outliers(data)
        data = self.transformer.remove_missing_data(data)
        data = self.transformer.apply_usage_correspondence(data)

        x_train, x_test, y_train, y_test = self.transformer.split_data(data)
        return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    data_conso = pd.read_csv("data/cleaned/data_extract.csv") # A mettre en maj -> var globale
