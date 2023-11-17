# Projet-4 - Anticipez les besoins en consommation de bâtiments

Ce projet a été réalisé dans le cadre de la formation diplomante de Data Scientist d'OpenClassRooms & CentraleSupelec.

## A propos du projet : 

### Objectifs : 
- Développer un modèle de prédiction de consommation énergétique des bâtiments pour la ville de Seattle.
<img width="1412" alt="Capture d’écran 2023-11-03 à 20 21 11" src="https://github.com/Emeline2104/Projet5_TAPIN_Final/assets/133622119/1fc0932a-5e22-40e4-a3fb-6fd104119b4e">

### Données : 
- Les données sont disponibles [ici](https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy).
- Elles incluent des relevés de la ville de 2016 (données structurelles des bâtiments (taille, usage, date de construction, etc.). 
  
### Méthodologie : 
1. Analyse exploratoire des données.
2. Exploration des différentes méthodes de pré-traitement des données (gestion de la qualité des données et feature engineering) et des modèles de regression (RL, Random Forest, Gradient Boosting, SVR) pour sélectionner le modèle optimal.
<img width="794" alt="Capture d’écran 2023-11-06 à 13 45 37" src="https://github.com/Emeline2104/Projet5_TAPIN_Final/assets/133622119/ce7934de-5f5a-4b50-b973-e75437df8fb4">
3. Mise en place de pipelines et pré-traitement et de modèlé de prédiction des consommations énergétiques des batiments.

### Livrables : 

#### Notebooks :
- Notebook de l'analyse exploratoire et de l'analyse de la qualité des données ([1_EDA.ipynb](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/notebooks/1_EDA.ipynb))
- Notebook exploratoire des méthodes utilisées (features engineering & modèles de prédiction) ([2_Prediction_consommation.ipynb](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/notebooks/2_Prediction_consommation.ipynb))
  
#### Scripts : 
- Script principal du projet (*main.py*) qui effectue les étapes suivantes :
  - Chargement des données à partir du fichier spécifié dans le fichier de configuration (*[config.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/config.py)*); 
  - Nettoyage des données à l'aide d'un pipeline défini dans le module data_cleaning (*[data_cleaning.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/preprocessing/data_cleaning.py)*;.
  - Feature Engineering à l'aide d'un pipeline défini dans le module feature_engineering (*[feature_engineering.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/preprocessing/feature_engineering.py)*);
  - Entraînement et évaluation d'un modèle de régression baseline (régression linéaire (RL)) en utilisant le pipeline défini dans le module baseline_model (*[baseline_model.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/models/baseline_model.py)*);
  - Entraînement et évaluation d'un modèle XGBoost en utilisant le pipeline défini dans le module xgboost_model (*[xgboost_model.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/models/xgboost_model.py)*).
    
#### Support de présentation de l'analyse exploratoire pour la soutenance (*[3_Presentation](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/Presentation.pdf)*).

## Installation et exécution 

### Installation

Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`.

```bash
pip install -r requirements.txt
```

Le fichier setup.py est également inclus pour permettre l'installation et la distribution du projet en tant que package Python. Il spécifie les dépendances nécessaires pour le projet. Vous pouvez installer ces dépendances en exécutant la commande suivante :
```bash
pip install .
```

### Execution du script
Pour exécuter le script, assurez-vous d'avoir Python 3.11 ou supérieur installé et exécutez la commande suivante dans le terminal :

```bash
python main.py
```
Assurez-vous également de personnaliser les chemins et les paramètres dans le fichier [config.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/config.py) selon les besoins de votre projet.

