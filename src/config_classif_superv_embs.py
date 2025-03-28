import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def classify_bug_reports(
    csv_path,
    title_col='Title',
    description_col='Description',
    classification_col='Classification',
    sep=','
):
    """
    Clasifica reportes de bugs usando embeddings de texto y varios modelos de machine learning.

    Parámetros:
    - csv_path (str): Ruta del archivo CSV que contiene los datos.
    - title_col (str): Nombre de la columna que contiene el título (default: 'Title').
    - description_col (str): Nombre de la columna que contiene la descripción (default: 'Description').
    - classification_col (str): Nombre de la columna que contiene la clasificación (default: 'Classification').

    Retorna:
    - pd.DataFrame: DataFrame con los resultados de la evaluación de los modelos.
    """
    # Iniciar tiempo de ejecución total
    start_time_total = time.time()

    print("="*80)
    print("INICIO DEL PROCESO DE CLASIFICACIÓN DE REPORTES DE BUGS")
    print("="*80)

    # Cargar el dataset
    print("\n[1/6] Cargando dataset...")
    load_start = time.time()
    df = pd.read_csv(csv_path, sep=sep)
    load_time = time.time() - load_start
    print(f"✓ Dataset cargado en {load_time:.2f} segundos")
    print(f"• Número total de reportes: {len(df)}")
    print(f"• Columnas disponibles: {', '.join(df.columns)}")

    # Preparación de datos
    print("\n[2/6] Preparando datos...")
    prep_start = time.time()
    print("• Concatenando título y descripción...")
    df['text'] = df[title_col].fillna('') + ' ' + df[description_col].fillna('')
    print("• Filtrando reportes sin clasificación...")
    df = df[df[classification_col].notna() & (df[classification_col] != '?')]
    print(f"• Número de reportes válidos después del filtrado: {len(df)}")

    # Mostrar distribución de clases
    class_distribution = df[classification_col].value_counts()
    print("\n• Distribución de clases:")
    for clase, count in class_distribution.items():
        print(f"  - {clase}: {count} reportes ({count/len(df)*100:.1f}%)")

    # Dividir los datos
    print("\n• Dividiendo datos en conjuntos de entrenamiento (80%) y prueba (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df[classification_col], test_size=0.2, random_state=42, stratify=df[classification_col]
    )
    prep_time = time.time() - prep_start
    print(f"✓ Datos preparados en {prep_time:.2f} segundos")
    print(f"• Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"• Conjunto de prueba: {len(X_test)} muestras")

    # Usar embeddings con un modelo pequeño pero eficaz
    print("\n[3/6] Configurando modelo de embeddings...")
    model_name = 'paraphrase-MiniLM-L6-v2'  # Modelo ligero y rápido
    print(f"• Utilizando modelo de embeddings: {model_name}")
    model_start = time.time()
    embedding_model = SentenceTransformer(model_name)
    model_load_time = time.time() - model_start
    print(f"✓ Modelo cargado en {model_load_time:.2f} segundos")

    # Generar embeddings
    print("\n[4/6] Generando embeddings de texto...")
    embed_start = time.time()
    print("• Codificando conjunto de entrenamiento...")
    X_train_embeddings = embedding_model.encode(X_train.tolist(), show_progress_bar=True)
    print("• Codificando conjunto de prueba...")
    X_test_embeddings = embedding_model.encode(X_test.tolist(), show_progress_bar=True)
    total_embed_time = time.time() - embed_start
    print(f"✓ Embeddings generados en {total_embed_time:.2f} segundos")
    print(f"• Dimensiones del conjunto de entrenamiento: {X_train_embeddings.shape}")
    print(f"• Dimensiones del conjunto de prueba: {X_test_embeddings.shape}")

    # Preparar pipelines con y sin oversampling
    print("\n[5/6] Configurando y entrenando modelos de clasificación...")

    # Definir los clasificadores a usar en los experimentos
    classifiers = {
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'SVM': SVC(class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    }

    # DataFrame para almacenar los resultados
    results = pd.DataFrame(columns=['Classifier', 'SMOTE', 'Accuracy', 'Macro F1', 'Clase 0 F1', 'Clase 1 F1', 'Proporción Clase Configuration', 'Error Proporción'])

    # Repetir experimentos para cada clasificador
    for clf_name, clf in classifiers.items():
        print(f"\nExperimentos con {clf_name}")

        # Experimento 1: Con SMOTE
        print(f"\nExperimento 1: {clf_name} con balanceo de clases usando SMOTE")
        pipeline_with_smote = ImbPipeline([
            ('sampler', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        # Experimento 2: Sin SMOTE
        print(f"\nExperimento 2: {clf_name} sin balanceo de clases (sin SMOTE)")
        pipeline_without_smote = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        # Entrenar ambos modelos
        print("\n• Iniciando entrenamiento de los modelos...")

        # Entrenar modelo con SMOTE
        print(f"\n[Experimento 1] Entrenando {clf_name} con SMOTE...")
        train_start_smote = time.time()
        pipeline_with_smote.fit(X_train_embeddings, y_train)
        train_time_smote = time.time() - train_start_smote
        print(f"✓ Modelo con SMOTE entrenado en {train_time_smote:.2f} segundos")

        # Entrenar modelo sin SMOTE
        print(f"\n[Experimento 2] Entrenando {clf_name} sin SMOTE...")
        train_start_no_smote = time.time()
        pipeline_without_smote.fit(X_train_embeddings, y_train)
        train_time_no_smote = time.time() - train_start_no_smote
        print(f"✓ Modelo sin SMOTE entrenado en {train_time_no_smote:.2f} segundos")

        # Evaluación
        print("\n[6/6] Evaluando los modelos...")

        # Evaluar modelo con SMOTE
        print(f"\n[Experimento 1] Evaluando {clf_name} con SMOTE...")
        eval_start_smote = time.time()
        y_pred_smote = pipeline_with_smote.predict(X_test_embeddings)
        accuracy_smote = accuracy_score(y_test, y_pred_smote)
        macro_f1_smote = f1_score(y_test, y_pred_smote, average='macro')
        f1_scores_smote = f1_score(y_test, y_pred_smote, average=None)
        f1_smote_clase0 = f1_scores_smote[0] if len(f1_scores_smote) > 0 else 0
        f1_smote_clase1 = f1_scores_smote[1] if len(f1_scores_smote) > 1 else 0
        eval_time_smote = time.time() - eval_start_smote
        print(f"✓ Evaluación completada en {eval_time_smote:.2f} segundos")

        # Calcular proporción de la clase "Configuration" y error
        prop_clase_config_smote = (y_pred_smote == 'Configuration').mean()
        prop_real_clase_config = (y_test == 'Configuration').mean()
        error_prop_clase_config_smote = abs(prop_clase_config_smote - prop_real_clase_config)

        # Evaluar modelo sin SMOTE
        print(f"\n[Experimento 2] Evaluando {clf_name} sin SMOTE...")
        eval_start_no_smote = time.time()
        y_pred_no_smote = pipeline_without_smote.predict(X_test_embeddings)
        accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
        macro_f1_no_smote = f1_score(y_test, y_pred_no_smote, average='macro')
        f1_scores_no_smote = f1_score(y_test, y_pred_no_smote, average=None)
        f1_no_smote_clase0 = f1_scores_no_smote[0] if len(f1_scores_no_smote) > 0 else 0
        f1_no_smote_clase1 = f1_scores_no_smote[1] if len(f1_scores_no_smote) > 1 else 0
        eval_time_no_smote = time.time() - eval_start_no_smote
        print(f"✓ Evaluación completada en {eval_time_no_smote:.2f} segundos")

        # Calcular proporción de la clase "Configuration" y error
        prop_clase_config_no_smote = (y_pred_no_smote == 'Configuration').mean()
        error_prop_clase_config_no_smote = abs(prop_clase_config_no_smote - prop_real_clase_config)

        # Almacenar resultados usando pd.concat para compatibilidad futura
        new_results = pd.DataFrame([
            {
                'Classifier': clf_name,
                'SMOTE': 'Yes',
                'Accuracy': accuracy_smote,
                'Macro F1': macro_f1_smote,
                'Clase 0 F1': f1_smote_clase0,
                'Clase 1 F1': f1_smote_clase1,
                'Proporción Clase Configuration': prop_clase_config_smote,
                'Error Proporción': error_prop_clase_config_smote
            },
            {
                'Classifier': clf_name,
                'SMOTE': 'No',
                'Accuracy': accuracy_no_smote,
                'Macro F1': macro_f1_no_smote,
                'Clase 0 F1': f1_no_smote_clase0,
                'Clase 1 F1': f1_no_smote_clase1,
                'Proporción Clase Configuration': prop_clase_config_no_smote,
                'Error Proporción': error_prop_clase_config_no_smote
            }
        ])
        results = pd.concat([results, new_results], ignore_index=True)

    # Mostrar resultados comparativos
    print("\n" + "="*80)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*80)
    print(results)

    # Calcular tiempo total de ejecución
    total_time = time.time() - start_time_total
    print("\n" + "="*80)
    print(f"PROCESO COMPLETADO EN {total_time:.2f} SEGUNDOS")
    print("="*80)

    return results

if __name__ == '__main__':
    #classify_bug_reports('data/uvl_bug_reports.csv')



    #classify_bug_reports('data/dataset_conf_bug_report_v3.csv', title_col='Summary', description_col='Description', classification_col='Classification', sep=';')


    classify_bug_reports('data/NABATS_sampled_dataset.csv', title_col='Summary', description_col='Description', classification_col='Classification', sep=';')
