import pandas as pd
import os
import edit_distance_evaluator

# Ruta del fichero CSV
file_path = 'output/c22ab115d41ca5944f3f05ba41f4495b/results.csv'

# Leer el fichero CSV
df = pd.read_csv(file_path)

# Comprobar si todas las columnas de 'llm_pred' están rellenadas (no null o blank)
if df['llm_pred'].notnull().all() and (df['llm_pred'] != '').all():
    print("Todas las columnas de 'llm_pred' están rellenadas.")
else:
    print("Hay columnas de 'llm_pred' que no están rellenadas.")

# Dado un csv, agrupar los datos por numero de caracteres en 'description' y por cada grupo usar el evaluator para evaluar la columna 'llm_pred'
path = "../output/c22ab115d41ca5944f3f05ba41f4495b/results.csv"
print("El path actual es:", os.getcwd())
print("El path del fichero es:", path)
data = pd.read_csv("output/c22ab115d41ca5944f3f05ba41f4495b/results.csv", sep=",")
evaluator = edit_distance_evaluator.evaluate

# Definir el tamaño del grupo
group_size = 1000

# Agrupar los datos por el número de caracteres en 'description' en grupos de tamaño similar
grouped = df.groupby(df['Description'].str.len() // group_size)

# Evaluar cada grupo
for group_index, group in grouped:
    length_range = f"{group_index * group_size}-{(group_index + 1) * group_size - 1}"
    print(f"Evaluando grupo con descripciones de longitud {length_range}")
    report = evaluator(group['Classification'], group['llm_pred'])
    # Guardar el reporte en un fichero
    output_file = "output/c22ab115d41ca5944f3f05ba41f4495b/report_all_groups.txt"
    with open(output_file, 'a') as f:
        f.write(f"Reporte para grupo con descripciones de longitud {length_range}\n")
        f.write(str(report) + "\n\n")
report = evaluator(data['Classification'], data["llm_pred"])