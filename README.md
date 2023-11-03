# Projet-4 - Anticipez les besoins en consommation de bâtiments

Ce projet a été réalisé dans le cadre de la formation diplomante de Data Scientist d'OpenClassRooms & CentraleSupelec.

## A propos du projet : 

### Objectifs : 
- Développer un modèle de prédiction de consommation énergétique des bâtiments et d'émission de gaz à effet de serre (GES) pour la ville de Seattle.
![Uploading Capture d’écran 2023-11-03 à 20.21.11.png…]()

### Données : 
- Les données sont disponibles ici : https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy.
- Elles incluent des relevés de la ville de 2016 (données structurelles des bâtiments (taille, usage, date de construction, etc.). 
  
### Méthodologie : 
- Analyse exploratoire.
- Pré-traitement des données (gestion de la qualité des données et feature engineering).
- Modèles de regression (RL, Random Forest, Gradient Boosting, SVR) sur différentes méthodes de feature engineering pour sélectionner le modèle optimal.
- ![Uploading Capture d’écran 2023-11-03 à 20.21.41.png…]()

### Livrables : 
- Notebook de l'analyse exploratoire et de l'analyse de la qualité des données (*1_EDA.ipynb*).
- Notebooks pour chaque prédiction (émissions de CO2 et consommation d'énergie) (*2_Prediction_consommation.ipynb* & *3_Prediction_emissions.ipynb*).
- Support de présentation pour la soutenance (*4_Presentation*).

## Installation

Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`.

```bash
pip install -r requirements.txt
