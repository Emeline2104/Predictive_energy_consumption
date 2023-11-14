# Projet-4 - Anticipez les besoins en consommation de bâtiments

Ce projet a été réalisé dans le cadre de la formation diplomante de Data Scientist d'OpenClassRooms & CentraleSupelec.

## A propos du projet : 

### Objectifs : 
- Développer un modèle de prédiction de consommation énergétique des bâtiments pour la ville de Seattle.
<img width="1412" alt="Capture d’écran 2023-11-03 à 20 21 11" src="https://github.com/Emeline2104/Projet5_TAPIN_Final/assets/133622119/1fc0932a-5e22-40e4-a3fb-6fd104119b4e">

### Données : 
- Les données sont disponibles ici : https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy.
- Elles incluent des relevés de la ville de 2016 (données structurelles des bâtiments (taille, usage, date de construction, etc.). 
  
### Méthodologie : 
- Analyse exploratoire.
- Exploration des différentes méthodes de pré-traitement des données (gestion de la qualité des données et feature engineering) et des modèles de regression (RL, Random Forest, Gradient Boosting, SVR) sur différentes méthodes de feature engineering pour sélectionner le modèle optimal.
- Mise en place de pipelines et pré-traitement et de modèlé de prédiction des consommations énergétiques des batiments.
<img width="794" alt="Capture d’écran 2023-11-06 à 13 45 37" src="https://github.com/Emeline2104/Projet5_TAPIN_Final/assets/133622119/ce7934de-5f5a-4b50-b973-e75437df8fb4">

### Livrables : 

#### Notebooks :
- Notebook de l'analyse exploratoire et de l'analyse de la qualité des données (*1_EDA.ipynb*).
- Notebooks exploratoire des méthodes utilisées (features engineering & modèles de prédiction) (*2_Prediction_consommation.ipynb*).
  
#### Scripts : 
- Script principal du projet (*main.py*) qui effectue les étapes suivantes :
  - Chargement des données à partir du fichier spécifié dans le fichier de configuration (*config.py*); 
  - Nettoyage des données à l'aide d'un pipeline défini dans le module data_cleaning (*data_cleaning.py*);.
  - Feature Engineering à l'aide d'un pipeline défini dans le module feature_engineering (*feature_engineering.py*);
  - Entraînement et évaluation d'un modèle de régression baseline (régression linéaire (RL)) en utilisant le pipeline défini dans le module baseline_model (*baseline_model.py*);
  - Entraînement et évaluation d'un modèle XGBoost en utilisant le pipeline défini dans le module xgboost_model (*xgboost_model.py*).
    
#### Support de présentation de l'analyse exploratoire pour la soutenance (*Presentation*).

## Installation

Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`.

```bash
pip install -r requirements.txt

Pour exécuter le script, assurez-vous d'avoir Python 3.11 ou supérieur installé et exécutez la commande suivante dans le terminal :

```bash
python main.py

Assurez-vous également de personnaliser les chemins et les paramètres dans le fichier config.py selon les besoins de votre projet.
