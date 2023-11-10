"""
Fichier : data_cleaning.py

Ce fichier regroupe des classes pour le nettoyage de données.


Classes :
- Cleaning : Regroupe des méthodes spécifiques de nettoyage de données.
Méthodes : extract_building_types, drop_outliers, remove_missing_data, apply_usage_correspondence,
split_data
- DataCleaningPipeline : Pipeline de nettoyage de données avec une étape de nettoyage 
de type Cleaning.
Méthode : clean.

Utilisation :
Charger les données à nettoyer depuis un fichier CSV.
Créer une instance de Cleaning avec ces données.
Créer une instance de DataCleaningPipeline avec l'instance de Cleaning.
Exécuter le nettoyage en appelant la méthode clean de la classe DataCleaningPipeline.
Afficher les données nettoyées.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

class Cleaning:
    """
    Regroupe les méthodes de nettoyage spécifiques, comme l'extraction des types de bâtiments,
    la suppression des valeurs aberrantes, la suppression des données manquantes, 
    et l'application de la correspondance des usages.
    Chaque méthode effectue une tâche spécifique de nettoyage.
    """

    def __init__(self, data):
        """
        Initialise l'instance de Cleaning avec les données à nettoyer.

        Parameters:
        - data (pd.DataFrame): Le DataFrame contenant les données à nettoyer.
        """
        self.data = data
        # Initialisation des attributs vide de split data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def extract_building_types(self):
        """
        Extrait les données des types de bâtiments spécifiques pour garder les bâtiments 
        non résidentiels.

        Returns:
        pandas.DataFrame: Le DataFrame filtré selon les critères spécifiés.
        """
        # Extraction des bâtiments non résidentiels
        building_types_to_keep = ['NonResidential', 'Nonresidential WA', 'Nonresidential COS']
        self.data = self.data[self.data['BuildingType'].isin(building_types_to_keep)]

        # Suppression de données spécifiques
        data_to_remove = [
                        'Other - Lodging/Residential',
                        'Residence Hall/Dormitory',
                        'Residential Care Facility',
                        'Multifamily Housing',
                        ]
        mask = self.data['LargestPropertyUseType'].isin(data_to_remove)
        self.data = self.data[~mask]

    def drop_outliers(self):
        """
        Supprime les valeurs aberrantes du DataFrame selon la colonne 'Outlier' 
        et la 'ComplianceStatus'.

        Returns:
        pandas.DataFrame: Le DataFrame sans les valeurs aberrantes.
        """
        self.data = self.data[
            (~self.data['Outlier'].isin(['High outlier', 'Low outlier']))
            &
            (self.data['ComplianceStatus'] == 'Compliant')
            ]

    def remove_missing_data(self):
        """
        Supprime les lignes contenant des données manquantes dans un DataFrame.

        Returns:
        pd.DataFrame: Le DataFrame nettoyé.
        """
        self.data = self.data.dropna(
            axis=0,
            how='any',
            subset=['LargestPropertyUseType', 'SiteEnergyUse(kBtu)']
            )

    def apply_usage_correspondence(self):
        """
        Applique une table de correspondance des usages à un DataFrame pour mapper les valeurs.

        Returns:
        pd.DataFrame: Le DataFrame avec les colonnes d'usage mappées selon la correspondance.
        """
        # Mapping des usages selon une table de correspondance
        correspondence_usage = {# faire fichier JSON ?
            'Data Center': 'Data Center',
            'Urgent Care/Clinic/Other Outpatient': 'Médical',
            'Laboratory': 'Médical',
            'Hospital (General Medical & Surgical)': 'Médical',
            'Supermarket/Grocery Store': 'Magasin',
            'Restaurant': 'Restaurant',
            'Other/Specialty Hospital' : 'Médical',
            'Museum' : 'Divertissement', 
            'Other - Restaurant/Bar' : 'Restaurant',
            'Other - Recreation' : 'Divertissement',
            'Police Station' :'Service public',
            'Parking' :'Parking',
            'Lifestyle Center' :'Sport', 
            'Senior Care Community' :'Service',
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
            'Retail Store' : 'Magasin', 
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
            self.data[final_usages[i]] = self.data[col].map(correspondence_usage)

    def split_data(self):
        """
        Prépare les données en séparant les features et la target, effectue la stratification et renvoie les ensembles d'entraînement et de test.

        Returns:
        X_train (pd.DataFrame): Ensemble d'entraînement des features.
        X_test (pd.DataFrame): Ensemble de test des features.
        y_train (pd.DataFrame): Ensemble d'entraînement de la target.
        y_test (pd.DataFrame): Ensemble de test de la target.
        """
        indic_conso = ['SiteEnergyUse(kBtu)']

        # Crée une table de features en supprimant les colonnes d'indicateurs de consommation
        features = self.data.drop(columns=indic_conso)
            
        # Crée une table de target en incluant uniquement les colonnes d'indicateurs de consommation
        target = self.data[indic_conso]
            
        # Convertit la colonne 'TypeUtilisationPrincipale' en chaîne de caractères
        features['TypeUtilisationPrincipale'] = features['TypeUtilisationPrincipale'].astype(str)
            
        # Stratification sur la variable catégorielle pour obtenir X_train et X_test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=0.2, random_state=None, stratify=features['TypeUtilisationPrincipale']
        )
            
        print("Dimensions de l'ensemble d'entraînement :", self.X_train.shape, self.y_train.shape)
        print("Dimensions de l'ensemble de test :", self.X_test.shape, self.y_test.shape)
            
        # Conversion des arrays NumPy en DataFrames
        self.X_train = pd.DataFrame(self.X_train, columns=features.columns)
        self.X_test = pd.DataFrame(self.X_test, columns=features.columns)
        self.y_train = pd.DataFrame(self.y_train, columns=target.columns)
        self.y_test = pd.DataFrame(self.y_test, columns=target.columns)

class DataCleaningPipeline:
    """
    Classe représentant un pipeline de nettoyage de données.
    """

    def __init__(self, cleaning_instance):
        """
        Initialise le pipeline avec une instance de Cleaning.

        Parameters:
        - cleaning_instance (Cleaning): L'instance de Cleaning à utiliser pour le nettoyage.
        """
        self.cleaning_instance = cleaning_instance

    def clean(self):
        """
        Applique les étapes de nettoyage sur les données.

        Parameters:
        - data (pd.DataFrame): Le DataFrame contenant les données à nettoyer.

        Returns:
        pd.DataFrame: Le DataFrame nettoyé.
        """
        # Appel des méthodes de nettoyage
        self.cleaning_instance.extract_building_types()
        self.cleaning_instance.drop_outliers()
        self.cleaning_instance.remove_missing_data()
        self.cleaning_instance.apply_usage_correspondence()
        self.cleaning_instance.split_data()
        return self.cleaning_instance.X_train, self.cleaning_instance.X_test, self.cleaning_instance.y_train, self.cleaning_instance.y_test
    
# Utilisation
data_exemple = pd.read_csv(
    '/Users/beatricetapin/Documents/2023/Data Science/Projet 4/Projet_4_TAPIN/data/raw/2016_Building_Energy_Benchmarking.csv'
    )

# Création d'une instance de DataCleaningPipeline avec Cleaning comme étape
pipeline = DataCleaningPipeline(Cleaning(data_exemple))

# Nettoyage des données
X_train, X_test, y_train, y_test  = pipeline.clean()
print(X_train.head())