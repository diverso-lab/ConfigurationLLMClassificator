import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import edit_distance_evaluator
import user_prompt_factory
from GenerativeModelClient import GenerativeModelClient
import json

def llm_classification_experiment(csv_path, client, user_prompt_factory, output_csv_path, evaluator, true_column="class"):
    # If output_csv_path exists, we simply evaluate it => it would be interesting for the client to cache calls already made
    # with the same prompt, same model and same configuration, and not make calls to the API in its case    
    if not os.path.exists(output_csv_path):
        data = pd.read_csv(csv_path, sep=";")
        
        # For each instace on the csv
        progress_bar = tqdm(data.iterrows(), total=len(data))
        for index, row in progress_bar:
            user_prompt = user_prompt_factory(row)

            # response would be used for logging, not used right now
            response_text, response = client.generate(user_prompt, progress_bar=progress_bar)
            
            # We adde the predicted class to the instace
            data.loc[index, 'llm_pred'] = response_text

            # Save the file with the predictions
            # (in each iteration in case the process is interrupted we do not lose the predictions already made)
            data.to_csv(output_csv_path, index=False)
    else:
        data = pd.read_csv(output_csv_path)

    # Evaluate the predictions
    report = evaluator(data[true_column], data["llm_pred"])    
    return report

if __name__ == "__main__":
    # Call configuration => This should be in an experiment configuration file
    csv_path = "../data/dataset_configuration_bug_report.csv"
    output_csv_path = "../output/preds_Llama3-1-8b.csv"
    true_column = "Classification"
    model = "meta-llama-3.1-8b-instruct"
    user_prompt_factory = user_prompt_factory.get
    # If the system prompt is built dynamically based on some definition files or whatever, 
    # it should be shaped as a factory too
    system_prompt = """The user will provide information about a bug report. This information includes: Bug-ID, Project, Summary, Link and Enviroment. Classify the bug report into 'Configuration Bug Report' or 'Other', where the first class indicates that the bug report is related or about a configuration bug or issue and the second class indicates that the bug report is NOT RELATED to a configuration bug or issue, and is related to other matters such as database related bugs, request for addition, functional bugs,GUI-related bugs, Network bugs, Performance, security, etc... If the bug report is related to any of those themes, we will classify them as "Other". The first category(Configuration Bug Report) regards bugs concerned with building configuration files. Most of them are related to problems caused by (i) external libraries that should be updated or fixed and (ii) wrong directory or file paths in xml or manifest artifacts. As an example, the bug report shown below falls under this category because it is mainly related to a wrong usage of external dependencies that cause issues in the web model of the application. Understand that bug reports can be either 'Configuration Bug Report' or 'Other'. Reply ONLY with one of the two classes: 'Configuration Bug Report' or 'Other'. I insist, only those words, do not use more that 3 words. """
        
    client = GenerativeModelClient(model=model, system_prompt=system_prompt, max_tokens=10, temperature=0)
    evaluator = edit_distance_evaluator.evaluate
    
    # Execution
    report = llm_classification_experiment(csv_path, client, user_prompt_factory, output_csv_path, evaluator, true_column=true_column)
    
    # Results
    print(json.dumps(report, indent=4))
    