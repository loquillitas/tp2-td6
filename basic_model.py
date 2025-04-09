import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

# Mostrar todas las columnas y filas en pantalla
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar datos de entrenamiento
train_data = pd.read_csv("/Users/guille/Desktop/ditella/td6/tp2/spotify-skipped-track/competition_data.csv")

# Cargar datos de evaluación
eval_data = pd.read_csv("/Users/guille/Desktop/ditella/td6/tp2/spotify-skipped-track/submission.csv")

# Usar solo una fracción del dataset para entrenamiento rápido
train_data = train_data.sample(frac=1/10)

# Separar variable objetivo y features
y_train = train_data["TARGET"]
X_train = train_data.drop(columns=["TARGET"])

# Usar solo las columnas numéricas (incluyendo 'id')
X_train = X_train.select_dtypes(include='number')

# Liberar memoria
del train_data
gc.collect()

# Definir y entrenar modelo
cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
cls.fit(X_train, y_train)

# Asegurar que eval_data tenga las mismas columnas que X_train
eval_data = eval_data[X_train.columns]

# Predecir probabilidades
y_preds = cls.predict_proba(eval_data)[:, cls.classes_ == 1].squeeze()

# Crear archivo de salida
submission_df = pd.DataFrame({"ID": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("nuevo.csv", sep=",", index=False)
