from setuptools import setup, find_packages

setup(
    name='Projet_4',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'branca',
        'folium',
        'matplotlib',
        'missingno',
        'scikit-learn',
        'scipy',
        'seaborn',
        'xgboost',
    ],

    entry_points={
        'console_scripts': [
            'nom_script = scr.main:main',
        ],
    },
)
