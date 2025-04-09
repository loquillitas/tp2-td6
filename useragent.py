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
data = pd.read_csv('spotify-skipped-track/competition_data.csv')

# imprimo las columnas y sus tipos
print(data.dtypes)
print(data.columns)

# Eliminar columnas innecesarias --> tambien hay que borrar id (unnamed:0)
data.drop(columns=['spotify_track_uri', 'username'], inplace=True)

# grafico la distribucion de algunas columnas 
# user agent decrypted
plt.figure(figsize=(12, 6))
data['user_agent_decrypted'].value_counts().plot(kind='bar')
plt.title('Distribución de user_agent_decrypted')
plt.xlabel('user_agent_decrypted')
plt.ylabel('Frecuencia')
# plt.show()

# imprimo la cantidad de valores unicos 
# print(data['user_agent_decrypted'].nunique())

# imprimo la cantidad de user agents = unkown
# print(data['user_agent_decrypted'].isnull().sum())


# for categoria in data['user_agent_decrypted'].unique():
#     print()
#     print(categoria)
# -> vemos que hay muchos user agents distintos, pero en realidad son similares

def categorize_user_agent(ua):
    if pd.isnull(ua):
        return 'unknown'
    ua = ua.lower()
    if 'android' in ua:
        return 'android'
    elif 'iphone' in ua or 'ios' in ua:
        return 'ios'
    elif 'macintosh' in ua:
        return 'mac'
    elif 'windows' in ua:
        return 'windows'
    elif ua.isnumeric():
        return 'numeric_id'
    else:
        return 'other'

data['user_agent_grouped'] = data['user_agent_decrypted'].apply(categorize_user_agent)

# for categoria in data['user_agent_grouped'].unique():
#     print()
#     print(categoria)

print(data['user_agent_grouped'].nunique())

# pero tambien sabemos que 
# unknown --> 83% 
# [null] --> 16% 
# Other (659) --> 1%


# genero una columna que indique el tipo de dispositivo (iphone, pc, android, etc)
from urllib.parse import unquote

data['user_agent_decrypted'] = data['user_agent_decrypted'].apply(lambda x: unquote(x) if pd.notnull(x) else x)

def clasificar_plataforma(plat):
    if pd.isnull(plat):
        return 'unknown'

    plat = str(plat).lower()

    if any(x in plat for x in ['android', 'ios', 'iphone', 'ipad', 'mobile', 'phone']):
        return 'movil'
    elif any(x in plat for x in ['windows', 'mac', 'web', 'desktop']):
        return 'pc'
    elif any(x in plat for x in ['tv', 'xbox', 'ps', 'console']):
        return 'tv/consola'
    else:
        return 'otro'



data['tipo_dispositivo'] = data['user_agent_grouped'].apply(clasificar_plataforma)

print("debuger")
for categoria in data['tipo_dispositivo'].unique():
    print()
    print(categoria)

print(data['tipo_dispositivo'].value_counts())

# print(data['platform'].dropna().unique())
