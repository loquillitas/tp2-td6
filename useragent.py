import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# Imprimir todas las filas y columnas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar los datos
data = pd.read_csv('competition_data.csv')


# Eliminar columnas innecesarias --> tambien hay que borrar id (unnamed:0)
data.drop(columns=['spotify_track_uri', 'username'], inplace=True)



# imprimo la cantidad de valores unicos 
# print(data['user_agent_decrypted'].nunique())

# imprimo la cantidad de user agents = unkown
# print(data['user_agent_decrypted'].isnull().sum())


# for categoria in data['user_agent_decrypted'].unique():
#     print()
#     print(categoria)
# -> vemos que hay muchos user agents distintos, pero en realidad son similares

# def categorize_user_agent(ua):
#     if pd.isnull(ua):
#         return 'unknown'
#     ua = ua.lower()
#     if 'android' in ua:
#         return 'android'
#     elif 'iphone' in ua or 'ios' in ua:
#         return 'ios'
#     elif 'macintosh' in ua:
#         return 'mac'
#     elif 'windows' in ua:
#         return 'windows'
#     elif ua.isnumeric():
#         return 'numeric_id'
#     else:
#         return 'other'

# data['user_agent_grouped'] = data['user_agent_decrypted'].apply(categorize_user_agent)

# for categoria in data['user_agent_grouped'].unique():
#     print()
#     print(categoria)

# print(data['user_agent_grouped'].nunique())

# pero tambien sabemos que 
# unknown --> 83% 
# [null] --> 16% 
# Other (659) --> 1%


# genero una columna que indique el tipo de dispositivo (iphone, pc, android, etc)
from urllib.parse import unquote

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


print(data['tipo_dispositivo'].value_counts())

