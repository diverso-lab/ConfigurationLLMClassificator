import pandas as pd
import matplotlib.pyplot as plt
import os
import edit_distance_evaluator

# Ruta del fichero CSV
#file_path = 'output/c22ab115d41ca5944f3f05ba41f4495b/results.csv'
# Leer el fichero CSV
#df = pd.read_csv(file_path)

# Comprobar si todas las columnas de 'llm_pred' están rellenadas (no null o blank)
# if df['llm_pred'].notnull().all() and (df['llm_pred'] != '').all():
#     print("Todas las columnas de 'llm_pred' están rellenadas.")
# else:
#     print("Hay columnas de 'llm_pred' que no están rellenadas.")

#cargamos data\dataset_configuration_bug_report_updated.csv
data = pd.read_csv("data/dataset_conf_bug_report_v3.csv", sep=";",encoding='utf-8')
#mostrar el total de instancias cuya columna description es nula o texto vacia
null_or_empty_descriptions = data['Description'].isnull() | (data['Description'].str.strip() == '')
total_null_or_empty = null_or_empty_descriptions.sum()
print(f"Total de instancias con 'description' nula o vacía: {total_null_or_empty}")
 
#dibuja una grafica que muestre la distribución de numero de caracteres de la columna Description
#data['Description'].str.len().hist(bins=200)
# plt.xlabel("Número de caracteres")
# plt.ylabel("Número de instancias")
# plt.title("Distribución del número de caracteres en la columna 'Description'")
# plt.xlim(0, 500)
# plt.show()

# Dado un csv, agrupar los datos por numero de caracteres en 'description' y por cada grupo usar el evaluator para evaluar la columna 'llm_pred'
data = pd.read_csv("output/aaa5b434d3698140776bd9dc84071f22/results.csv", sep=",")
evaluator = edit_distance_evaluator.evaluate

# Definir el tamaño del grupo
#group_size = 1000
# Agrupar los datos por el número de caracteres en 'description' en grupos de tamaño similar
#grouped = df.groupby(df['Description'].str.len() // group_size)

# Agrupar los datos en grupos acumulativos basados en la longitud de 'description'
bins = [0, 50, 100, 150, 200, 250, 300, 400, 500,750, 1000, 2000, 5000, float('inf')]
labels = ["aa0-50", "ab0-100", "ac0-150", "ad0-200", "ae0-250", "af0-300", "ag0-400", "ah0-500","ai0-750", "aj0-1000", "ak0-2000", "al0-5000", "am0-inf"]
data['group'] = pd.cut(data['Description'].str.len(), bins=bins, labels=labels)

# Crear grupos acumulativos
grouped_data = []
for i in range(1, len(bins)):
    group = data[data['Description'].str.len() <= bins[i]]
    group['group'] = labels[i-1]
    grouped_data.append(group)

grouped = pd.concat(grouped_data).groupby('group')
f1_variances = []

#Delete file if exists to avoid appending
output_file = "output/aaa5b434d3698140776bd9dc84071f22/report_all_groups_acumulative.txt"
if os.path.exists(output_file):
    os.remove(output_file)

# Evaluar cada grupo
for group_index, group in grouped:
    print(f"Calculando resultados para la label {group_index}...")
    report = evaluator(group['Classification'], group['llm_pred'])
    print(report)
    f1_score = report['weighted avg']['f1-score']
    f1_variances.append((group_index, f1_score))
    # Guardar el reporte en un fichero
    output_file = "output/aaa5b434d3698140776bd9dc84071f22/report_all_groups_acumulative.txt"
    with open(output_file, 'a') as f:
        f.write(f"Reporte para grupo {group_index[2:]}. Num bug reports: {grouped.size()[group_index]}.\n")
        f.write(str(report) + "\n\n")

# Crear una gráfica de cómo varía el f1_score
group_labels, f1_scores = zip(*f1_variances)
group_labels = [label[2:] for label in group_labels]
plt.figure(figsize=(10, 6))
plt.plot(group_labels, f1_scores, marker='o')
plt.xticks(rotation=90)
plt.xlabel("Grupos acumulativos de longitud de 'Description'")
plt.ylabel("F1 Score")
plt.title("Variación del F1 Score según la longitud ACUMULATIVA de 'Description'")
plt.grid(True)
plt.show()

# Crear una gráfica que muestre el NUMERO DE MUESTRAS en cada grupo no acumulativo
group_sizes = data.groupby('group').size()
group_labels, sizes = zip(*group_sizes.items())
group_labels = [label[2:] for label in group_labels]

plt.figure(figsize=(10, 6))
plt.bar(group_labels, sizes)
plt.xticks(rotation=90)
plt.xlabel("Grupos de longitud de 'Description'")
plt.ylabel("Número de muestras")
plt.title("Número de muestras en cada grupo de longitud de 'Description'")
plt.grid(True)
plt.show()

#Ahora hacemos el estudio de derecha a izquierda (mayor grupo a menor grupo), agrupando los datos por el numero de caracteres en 'description' de mayor grupo (el conjunto total) al menor grupo (entre 0-50 caracteres)

# Agrupar los datos en grupos acumulativos basados en la longitud de 'description'
bins = [0, 50, 100, 150, 200, 250, 300, 400, 500,750, 1000, 2000, 5000, float('inf')]
labels = ["00inf-5000", "01inf-2000", "02inf-1000","03inf-750", "04inf-500", "05inf-400", "06inf-300", "07inf-250", "08inf-200", "09inf-150", "10inf-100", "11inf-50", "12total"]
data['group'] = pd.cut(data['Description'].str.len(), bins=bins, labels=labels)

# Crear grupos acumulativos en orden inverso
grouped_data = []
for i in range(len(bins) - 1, 0, -1):
    group = data[data['Description'].str.len() > bins[i-1]]
    group['group'] = labels[i-1]
    grouped_data.append(group)

grouped = pd.concat(grouped_data).groupby('group')
f1_variances = []

#Delete file if exists to avoid appending
output_file = "output/aaa5b434d3698140776bd9dc84071f22/report_all_groups_acumulative_inverse.txt"
if os.path.exists(output_file):
    os.remove(output_file)

# Evaluar cada grupo
for group_index, group in grouped:
    print(f"Calculando resultados para la label {group_index}...")
    report = evaluator(group['Classification'], group['llm_pred'])
    f1_score = report['weighted avg']['f1-score']
    f1_variances.append((group_index, f1_score))
    # Guardar el reporte en un fichero
    with open(output_file, 'a') as f:
        f.write(f"Reporte para grupo {group_index[2:]}. Num bug reports: {grouped.size()[group_index]}.\n")
        f.write(str(report) + "\n\n")

# Crear una gráfica de cómo varía el f1_score
group_labels, f1_scores = zip(*f1_variances)
group_labels = [label[2:] for label in group_labels]
plt.figure(figsize=(10, 6))
plt.plot(group_labels, f1_scores, marker='o')
plt.xticks(rotation=90)
plt.xlabel("Grupos acumulativos de longitud de 'Description'")
plt.ylabel("F1 Score")
plt.title("Variación del F1 Score según la longitud acumulativa de 'Description' (Orden Inverso)")
plt.grid(True)
plt.show()