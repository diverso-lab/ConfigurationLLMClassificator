import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import edit_distance_evaluator
import user_prompt_factory
from GenerativeModelClient import GenerativeModelClient
import json

def llm_classification_experiment(csv_path, client, user_prompt_factory, output_csv_path, evaluator, true_column="class"):    
    # Si output_csv_path existe, simplemente lo evaluamos => sería interesante que el cliente cacheara llamadas ya realizadas
    # con el mismo prompt, mismo modelo y misma configuración, y no hiciera las llamadas a la API en su caso
    if not os.path.exists(output_csv_path):
        data = pd.read_csv(csv_path, sep=";")
        
        # Para cada instancia del csv
        progress_bar = tqdm(data.iterrows(), total=len(data))
        for index, row in progress_bar:
            user_prompt = user_prompt_factory(row)

            # response se utilizaría para log, ahora mismo no se usa
            response_text, response = client.generate(user_prompt, progress_bar=progress_bar)
            
            # Añadimos la clase predicha a la instancia
            data.loc[index, 'llm_pred'] = response_text

            # Guardamos el fichero con predicciones
            # (en cada iteración por si se interrumpe el proceso no perder 
            # las predicciones ya realizadas)
            data.to_csv(output_csv_path, index=False)
    else:
        data = pd.read_csv(output_csv_path)

    # Evaluamos
    report = evaluator(data[true_column], data["llm_pred"])    
    return report

if __name__ == "__main__":
    # Configuración => Esto debería estar en un fichero de configuración del experimento
    csv_path = "../data/dataset_configuration_bug_report.csv"
    output_csv_path = "../output/preds_Llama3-1-8b.csv"
    true_column = "Classification"
    model = "meta-llama-3.1-8b-instruct"
    user_prompt_factory = user_prompt_factory.get
    # Si el system prompt se construye dinámicamente basándose en algunos archivos de definiciones o lo que
    # sea, habría que darle forma de factoría también
    system_prompt = """The user will provide information about a bug report. This information includes: Bug-ID, Project, Summary, Link and Enviroment. Classify the bug report into 'Configuration Bug Report' or 'Other', where the first class indicates that the bug report is related or about a configuration bug or issue and the second class indicates that the bug report is NOT RELATED to a configuration bug or issue, and is related to other matters such as database related bugs, request for addition, functional bugs,GUI-related bugs, Network bugs, Performance, security, etc... If the bug report is related to any of those themes, we will classify them as "Other". The first category(Configuration Bug Report) regards bugs concerned with building configuration files. Most of them are related to problems caused by (i) external libraries that should be updated or fixed and (ii) wrong directory or file paths in xml or manifest artifacts. As an example, the bug report shown below falls under this category because it is mainly related to a wrong usage of external dependencies that cause issues in the web model of the application. Understand that bug reports can be either 'Configuration Bug Report' or 'Other'. Reply ONLY with one of the two classes: 'Configuration Bug Report' or 'Other'. I insist, only those words, do not use more that 3 words. """
        
    client = GenerativeModelClient(model=model, system_prompt=system_prompt, max_tokens=10, temperature=0)
    evaluator = edit_distance_evaluator.evaluate
    
    # Ejecución
    report = llm_classification_experiment(csv_path, client, user_prompt_factory, output_csv_path, evaluator, true_column=true_column)
    
    # Resultados
    print(json.dumps(report, indent=4))
    