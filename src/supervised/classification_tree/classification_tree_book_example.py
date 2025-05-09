import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.datasets import get_rdataset
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
    (RandomForestRegressor as RF,
     GradientBoostingRegressor as GBR)
from ISLP.bart import BART

# transformer la VD/y en variable catégorielle à 2 valeurs (car elle est continue de base)
# Sales (continu) devient High (catégoriel)
# notre nouvelle VD = High
Carseats = load_data('Carseats')
High = np.where(Carseats.Sales > 8,
                "Yes",
                "No")

# prédire la VD High en utilisant toutes les AUTRES variables
# pour ce faire, matrice modèle
model = MS(Carseats.columns.drop('Sales'), intercept=False)
D = model.fit_transform(Carseats)
feature_names = list(D.columns)
X = np.asarray(D)

# maxdepth : quelle profondeur d'arbre on veut
# min_samples_split : nombre minimum d'observations dans un noeud pour être éligible pour le splitting
# criterion : le split criterion doit être Gini ou entropy
# random_state : pour la reproductibilité pcq si il y a égalité, le split cirterion est aléatoire
clf = DTC(criterion='entropy',
          max_depth=3,
          random_state=0)
clf.fit(X, High)
