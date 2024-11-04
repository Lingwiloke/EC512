import matplotlib as mpl
from matplotlib import pyplot as plt

import seaborn as sb
import pandas as pd

# utilisé ici pour numériser des catégories (encodage)
from sklearn import preprocessing

# nous chargeons un jeu de données
penguins = sb.load_dataset("penguins")
# nous pouvons visualiser ces données
penguins

# taille
penguins.shape

# recherche de données manquantes (fonction isnull)
penguins.isnull().sum()

# clés
penguins.keys()

# valeurs de l'index
penguins.index

# colonne 'ile'
penguins['island']

# ligne 12
penguins.loc[12]

# combien avons nous d'espèces de pingouins différentes ?
penguins['species'].unique()

# Quelle est la répartition des obserations par espèces ?
Species=penguins.species.value_counts()
Species.plot(kind='pie',autopct="%.2f%%");

# Quelques statistiques (arrondies à une decimale)
penguins.describe().round(1)

## Nous pouvons fixer les dimensions des différents graphiques que nous allons réaliser avec Seaborn
sb.set_theme()
sb.set(rc = {"figure.figsize": (12,8), "figure.dpi" : 100})

## avec seaborn
g = sb.scatterplot(x="bill_length_mm", y="bill_depth_mm", data=penguins, hue="species")
## ajout de titres et d'axes
g.get_legend().set_title("Espèces de pingouin") # modification du titre de la légende
plt.title("Longeur du bec en fonction de la hauteur")
plt.xlabel('Longueur du bec (mm)')
plt.ylabel('Hauteur du bec (mm)');

sb.histplot(x ="flipper_length_mm", data=penguins)
plt.title("Longueur des nageoires, toutes espèces confondues")
plt.xlabel('Longueur des nageoires (mm)')
plt.ylabel('Effectif');

# intervalle de taille 2
sb.histplot(x = "flipper_length_mm", data = penguins, binwidth=2)
plt.title("Longueur des nageoires, toutes espèces confondues")
plt.xlabel('Longueur des nageoires (mm)')
plt.ylabel('Effectif');

g = sb.histplot(x = "flipper_length_mm", data = penguins, binwidth=2, kde=False, hue = "species")
g.get_legend().set_title("Espèces de pingouin")
plt.title("Longueur des nageoires par espèces")
plt.xlabel('Longueur des nageoires (mm)')
plt.ylabel('Effectif');

sb.pairplot(data=penguins, hue = "species", height=3);

# Encodeur de données textuelles en entiers
label_encoder = preprocessing.LabelEncoder()

# Encodage (fonction fit_transform)
penguins['species'] = label_encoder.fit_transform(penguins['species'])
penguins["island"] = label_encoder.fit_transform(penguins["island"])
penguins["sex"] = label_encoder.fit_transform(penguins["sex"])

corr = penguins.corr()

sb.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, cmap= 'crest');

