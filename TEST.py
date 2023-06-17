import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
#import math
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

# TITRE
st.title("ESTIMATION DU PRIX DE LA MAISON")

# Pour permettre à l'application de sauvegarder notre data et eviter qu'elle ne se recharge à chaque excution, ce qui permet à l'app d'etre plus rapide
st.cache(persist=True)

#Preprocessing - Préparation des données
#***************************************

train= pd.read_csv("C:\\Users\\MSI LEOPARD PRO\\Documents\\APP_MAISON\\Data\\files\\train.csv")
test= pd.read_csv("C:\\Users\\MSI LEOPARD PRO\\Documents\\APP_MAISON\\Data\\files\\test.csv")
sample_submission= pd.read_csv("C:\\Users\\MSI LEOPARD PRO\\Documents\\APP_MAISON\\Data\\files\\sample_submission.csv")

#création d'une copie de chaque dataset
test_copy  = test.copy()
train_copy  = train.copy()

test_copy.head()
train_copy.head()

train_copy['train']  = 1
test_copy['train']  = 0
data_full = pd.concat([train_copy, test_copy], axis=0,sort=False)

## data_full.describe()

## data_full.info()

df_NULL = [(c, data_full[c].isna().mean()*100) for c in data_full]
df_NULL = pd.DataFrame(df_NULL, columns=["Colonne", "Taux de NULL"])
df_NULL.sort_values("Taux de NULL", ascending=False)

# Variables avec plus de 50% de NULL
df_NULL = df_NULL[df_NULL["Taux de NULL"] > 80]
df_NULL.sort_values("Taux de NULL", ascending=False)

list_NULL_features = list(df_NULL.Colonne)
data_full = data_full.drop(list_NULL_features,axis=1)

#Features engineering
#********************

categorical_features = data_full.select_dtypes(include=['object'])
numerical_features = data_full.select_dtypes(exclude=['object'])

# Variables numériques :
print("Nombre de variables numériques :",numerical_features.shape[1])
print("\nNombre de valeurs nulles :\n",numerical_features.isnull().sum())

print("Nombre de variables numériques :",categorical_features.shape[1])
print("\nNombre de valeurs nulles :\n",categorical_features.isnull().sum())

fill_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
categorical_features[fill_None]= categorical_features[fill_None].fillna('None')

fill_other = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']
categorical_features[fill_other] = categorical_features[fill_other].fillna(categorical_features.mode().iloc[0])

categorical_features.info()

print("Médiane GarageYrBlt :",numerical_features['GarageYrBlt'].median())
print("LotFrontage :",numerical_features["LotFrontage"].median())

numerical_features['GarageYrBlt'] = numerical_features['GarageYrBlt'].fillna(numerical_features['GarageYrBlt'].median())
numerical_features['LotFrontage'] = numerical_features['LotFrontage'].fillna(numerical_features['LotFrontage'].median())

numerical_features = numerical_features.fillna(0)
numerical_features.info()

for col in categorical_features.columns:
    #Conversion du type de variable en variable catégorique
    categorical_features[col] = categorical_features[col].astype('category')
    categorical_features[col] = categorical_features[col].cat.codes
categorical_features.head()

df_final = pd.concat([numerical_features,categorical_features], axis=1,sort=False)
final_train = df_final[df_final['train'] == 1]
final_train = final_train.drop(['train',],axis=1)

final_test = df_final[df_final['train'] == 0]
final_test = final_test.drop(['SalePrice'],axis=1)
final_test = final_test.drop(['train',],axis=1)

#Features selection - Analyse des correlations
#*********************************************

final_train = final_train.drop(["Id"],axis=1)

corr_train = final_train.corr()

# Masque pour la partie haute du Heatmap
mask = np.triu(np.ones_like(corr_train, dtype=bool))
# Création de la heatmap Seaborn
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_train, mask=mask, cmap="coolwarm", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_title("Matrice des corrélations du jeu de données Train", fontsize=22)
plt.show()

nocorr_features = list(corr_train[corr_train['SalePrice']<0.2].index)
nocorr_features

final_train = final_train.drop(nocorr_features, axis=1)
final_train.head()

#Modèlisation par régression linéaire
#************************************

rl_features = list(corr_train[corr_train['SalePrice']>0.3].index)
rl_features.remove("SalePrice")
rl_features

Y_train = final_train["SalePrice"]
X_train = final_train.drop(["SalePrice"],axis=1)
X_train = X_train[rl_features]

#Split des données
#*****************

from sklearn.model_selection import train_test_split
X_train_rl, X_test_rl, y_train_rl, y_test_rl = train_test_split(X_train, Y_train, test_size=0.3, random_state=1)

y_train_rl = y_train_rl.values.reshape(-1,1)
y_test_rl = y_test_rl.values.reshape(-1,1)

#Standardisation des données
#***************************

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_rl = sc.fit_transform(X_train_rl)
X_test_rl = sc.fit_transform(X_test_rl)
y_train_rl = sc.fit_transform(y_train_rl)
y_test_rl = sc.fit_transform(y_test_rl)

#Première régression linéaire
#****************************

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train_rl,y_train_rl)

print("Intercept :",lm.intercept_)
print("Coefficients :",lm.coef_)
print("R² du modèle :",round(lm.score(X_train_rl,y_train_rl),2))

pred_rl = lm.predict(X_test_rl)
pred_rl = pred_rl.reshape(-1,1)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(y_test_rl, pred_rl)
ax.plot([y_test_rl.min(), y_test_rl.max()], [y_test_rl.min(), y_test_rl.max()], color='r')
ax.set(xlabel='y_test', ylabel='y_pred')
plt.title("Projection des prédictions en fonction des valeurs réelles", fontsize=20)
plt.show()

#Calcul des métriques de performances
#************************************

#Fonction de calculs des metriques importantes MAE, MSE, MAPE, RMSE
def metrics_timeseries(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = y_true - y_pred
    mae = np.mean(abs(diff))
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(diff / y_true)) * 100
    dict_metrics = {"Métrique":["MAE", "MSE", "RMSE", "MAPE"], "Résultats":[mae, mse, rmse, mape]}
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics

metrics_rl = metrics_timeseries(y_test_rl, pred_rl)
metrics_rl

#Modèlisation par Random Forest
#*******************************

Y_rf = final_train["SalePrice"]
X_rf = final_train.drop(["SalePrice"],axis=1)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
selector = RFECV(RandomForestRegressor(), min_features_to_select=5, step=1, cv=5)
selector.fit(X_rf,Y_rf)

selector.grid_scores_

selector.ranking_

best_features_rf = list(np.array(X_rf.columns)[selector.support_])
best_features_rf

X_rf = X_rf[best_features_rf]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, Y_rf, test_size=0.3, random_state=1)
y_train_rf = y_train_rf.values.reshape(-1,1)
y_test_rf = y_test_rf.values.reshape(-1,1)

#Utilisation de GridSearchCV pour sélectionner les meilleurs paramètres
#**********************************************************************

from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
param_grid_rf = { 'n_estimators' : [10,50,100,150,200], 'max_features' : ['auto', 'sqrt']}
grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5)
grid_search_rf.fit(X_train_rf, y_train_rf)

print ("Score final : ", round(grid_search_rf.score(X_train_rf, y_train_rf) *100,4), " %")
print ("Meilleurs parametres: ", grid_search_rf.best_params_)
print ("Meilleure config: ", grid_search_rf.best_estimator_)

rf =  RandomForestRegressor(max_features='sqrt', n_estimators=150)
rf.fit(X_train_rf, y_train_rf)

pred_rf = rf.predict(X_test_rf)
pred_rf = pred_rf.reshape(-1,1)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(y_test_rf, pred_rf)
ax.plot([y_test_rf.min(), y_test_rf.max()], [y_test_rf.min(), y_test_rf.max()], color='r')
ax.set(xlabel='y_test', ylabel='y_pred')
plt.title("Projection des prédictions en fonction des valeurs réelles", fontsize=20)
plt.show()

metrics_rf = metrics_timeseries(y_test_rf, pred_rf)
metrics_rf

# Afin de comparer les métriques, on inverse la standardisation de la régression linéaire
metrics_rl_i = metrics_timeseries(sc.inverse_transform(y_test_rl), sc.inverse_transform(pred_rl))
metrics_rl_i

#PREDICTION DU PRIX DES MAISONS DU FICHIER SAMPLE SUBMISSION
#***********************************************************

id_test = final_test["Id"]
X_pred_test = final_test[best_features_rf]

pred_rf = rf.predict(X_pred_test)
pred_rf = pred_rf.reshape(-1,1)
pred_rf

df_submission = pd.concat([id_test,pd.Series(pred_rf[:,0])],axis=1).rename(columns={0:"SalePrice"})
df_submission







