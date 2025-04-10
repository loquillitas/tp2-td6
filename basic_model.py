import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV 
import xgboost as xgb
from urllib.parse import unquote


# Mostrar todas las columnas y filas en pantalla
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar datos 
data = pd.read_csv("competition_data.csv")

# PROCESAMIENTO DE DATOS 
# Eliminar columnas innecesarias --> ¿tambien hay que borrar id (unnamed:0)?
data.drop(columns=['spotify_track_uri', 'username'], inplace=True)


# Quiero ver el TIPO DE DISPOSITIVO desde el que se inicio sesion --> se encuentran en platform
# decodificamos los caracteres especiales para 'platform'
data['user_agent_decrypted'] = data['platform'].apply(lambda x: unquote(x) if pd.notnull(x) else x) 
# paso a minusculas
data['user_agent_decrypted'] = data['platform'].str.lower()

def clasificar_plataforma(plat:str) -> str:
    """
    clasifica el dispositivo (celular, pc, etc) a partir del atributo plataforma, pasado por parametro
    veo si dentro de plat existe un str tipo 'ios', 'iphone', etc
    esta funcion va a generar 5 posibles categorias (output) para la columna 'tipo_dispositivo'
    """

    if pd.isnull(plat):
        return 'unknown'    #1
    if any(x in plat for x in ['android', 'ios', 'iphone', 'ipad', 'mobile', 'phone']):
        return 'movil'      #2
    elif any(x in plat for x in ['windows', 'mac', 'web', 'desktop']):
        return 'pc'         #3
    elif any(x in plat for x in ['tv', 'xbox', 'ps', 'console']):
        return 'tv/consola' #4
    else:
        return 'otro'       #5

data['tipo_dispositivo'] = data['platform'].apply(clasificar_plataforma)
# Eliminar columnas innecesarias
data.drop(columns=['platform',
                    'user_agent_decrypted',  
                    'master_metadata_track_name', 
                    'master_metadata_album_artist_name',
                    'master_metadata_album_album_name',
                    'ts'], inplace=True)


# TRAIN SET
train_data = data.sample(frac=0.8, random_state=42)
## Usar solo una fracción del dataset para entrenamiento rápido
train_data = train_data.sample(frac=6/10)
# Separar variable objetivo y features
y_train = train_data["TARGET"]
x_train = train_data.drop(columns=["TARGET"])
# Usar solo las columnas numéricas (incluyendo 'id') y booleanas

x_train = x_train.select_dtypes(include=['number', 'bool'])

# VAL SET
val_data = data.drop(train_data.index)
# Separar variable objetivo y features
y_val = val_data["TARGET"]
x_val = val_data.drop(columns=["TARGET"])
# Usar solo las columnas numéricas (incluyendo 'id')
x_val = x_val.select_dtypes(include=['number', 'bool'])

# Cargar datos de evaluación
eval_data = pd.read_csv("submission.csv")


# Liberar memoria
del train_data
gc.collect()

# Definir y entrenar modelo --> usamos grid search
## defino el grid con los hiperparámetros
param_grid = {
    # 'criterion': ['gini', 'entropy'],   # Criterio para dividir el árbol
    'splitter': ['best'],     # Estrategia de división
    # 'max_features': ['auto', 'sqrt'],   # Número máximo de características a considerar para la división
    'class_weight': ['balanced'], # Manejo de clases desbalanceadas
    # 'max_leaf_nodes': [None, 10, 20],   # Número máximo de nodos hoja
    # 'min_weight_fraction_leaf': [0.0, 0.1], # Fracción mínima de peso para crear un nodo hoja
    # 'max_samples': [0.5, 0.8],          # Fracción de muestras a usar para entrenar cada árbol
    # 'max_features': [0.45, 0.5],         # Fracción de características a usar para entrenar cada árbol
    # 'min_impurity_decrease': [0.0, 0.1], # Reducción mínima de impureza para dividir un nodo
    'max_depth': [8, 10, 12, 14],         # Profundidad máxima del árbol
    'min_samples_split': [2, 5],    # Número mínimo de muestras para dividir un nodo
    'min_samples_leaf': [5, 7, 9]       # Número mínimo de muestras para crear un nodo hoja
    
}

# Definir el modelo
modelo = DecisionTreeClassifier(random_state=42)

grid = GridSearchCV(modelo, param_grid, cv=5, scoring='roc_auc', n_jobs=1)
grid.fit(x_train, y_train)

# Imprimir los mejores hiperparámetros
print("Mejores hiperparámetros:")
print(grid.best_params_)


# Obtener el mejor modelo
modelo = grid.best_estimator_

# calculo AUC-ROC
y_preds = modelo.predict_proba(x_val)[:, modelo.classes_ == 1].squeeze()
auc = roc_auc_score(y_val, y_preds)
print(f"AUC-ROC: {auc:.4f}")


# Asegurar que eval_data tenga las mismas columnas que X_train
eval_data = eval_data[x_train.columns]

# Predecir probabilidades
y_preds = modelo.predict_proba(eval_data)[:, modelo.classes_ == 1].squeeze()

# # Crear archivo de salida
submission_df = pd.DataFrame({"ID": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["ID"].astype(int)
submission_df.to_csv("nuevo.csv", sep=",", index=False)
