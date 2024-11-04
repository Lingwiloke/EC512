# %%
# Visualiser des données
# Nous avosn ici le code seul sans les explication qui se trouvent dans le notebook

# %%
# import des bibliothèques
import matplotlib as mpl
from matplotlib import pyplot as plt

import seaborn as sb
import pandas as pd

# utilisé ici pour numériser des catégories (encodage)
from sklearn import preprocessing

# %%
# nous chargeons un jeu de données
penguins = sb.load_dataset("penguins")
# nous pouvons visualiser ces données
print("Données \n",penguins,"\n\n")

# taille
taille = penguins.shape
print("Taille \n",taille,"\n\n")

# recherche de données manquantes (fonction isnull)
donnes_nulles = penguins.isnull().sum()
print("Données manquantes \n",donnes_nulles,"\n\n")

# clés
cles = penguins.keys()
print("Clés \n",cles,"\n\n")

# valeurs de l'index
Index = penguins.index
print("Index \n",Index,"\n\n")

# colonne 'ile'
Ile = penguins['island']
print("Colonne île \n",Ile,"\n\n")

# ligne 12
L12 = penguins.loc[12]
print("Ligne 12 \n",L12,"\n\n")

# combien avons nous d'espèces de pingouins différentes ?
Especes = penguins['species'].unique()
print("Combien avons nous d'espèces de pingouins différentes ? \n",Especes,"\n\n")

# %%
# Quelle est la répartition des obserations par espèces ?
Species=penguins.species.value_counts()
Species.plot(kind='pie',autopct="%.2f%%");
plt.show()

# %%
# Quelques statistiques (arrondies à une decimale)
penguins.describe().round(1)

# %%
## Nous pouvons fixer les dimensions des différents graphiques que nous allons réaliser avec Seaborn
sb.set_theme()
sb.set(rc = {"figure.figsize": (12,8), "figure.dpi" : 100})

# %%
## avec seaborn
f1 = sb.scatterplot(x="bill_length_mm", y="bill_depth_mm", data=penguins, hue="species")
## ajout de titres et d'axes
f1.get_legend().set_title("Espèces de pingouin") # modification du titre de la légende
plt.title("Longeur du bec en fonction de la hauteur")
plt.xlabel('Longueur du bec (mm)')
plt.ylabel('Hauteur du bec (mm)');
plt.show()

# %%
f2 = sb.histplot(x ="flipper_length_mm", data=penguins)
plt.title("Longueur des nageoires, toutes espèces confondues")
plt.xlabel('Longueur des nageoires (mm)')
plt.ylabel('Effectif');
plt.show()

# intervalle de taille 2
f3 = sb.histplot(x = "flipper_length_mm", data = penguins, binwidth=2)
plt.title("Longueur des nageoires, toutes espèces confondues")
plt.xlabel('Longueur des nageoires (mm)')
plt.ylabel('Effectif');
plt.show()

f4 = sb.histplot(x = "flipper_length_mm", data = penguins, binwidth=2, kde=False, hue = "species")
f4.get_legend().set_title("Espèces de pingouin")
plt.title("Longueur des nageoires par espèces")
plt.xlabel('Longueur des nageoires (mm)')
plt.ylabel('Effectif');
plt.show()

# %%
f5 = sb.pairplot(data=penguins, hue = "species", height=3);
plt.show()

# %%
# Encodeur de données textuelles en entiers
label_encoder = preprocessing.LabelEncoder()

# Encodage (fonction fit_transform)
penguins['species'] = label_encoder.fit_transform(penguins['species'])
penguins["island"] = label_encoder.fit_transform(penguins["island"])
penguins["sex"] = label_encoder.fit_transform(penguins["sex"])

corr = penguins.corr()

sb.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, cmap= 'crest');
plt.show()

