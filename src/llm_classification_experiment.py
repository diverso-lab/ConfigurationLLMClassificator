import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import edit_distance_evaluator
import user_prompt_factory
from GenerativeModelClient import GenerativeModelClient
import json
import hashlib

def compute_hash(config):
    #Compute a unique hash for the experiment configuration.
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def save_experiment_results(output_dir, config, data, report=None):
    #Save the experiment configuration, results, and report to files.
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Save results
    results_path = os.path.join(output_dir, "results.csv")
    data.to_csv(results_path, index=False)
    
    if report is not None:
        # Save report
        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

def load_experiment_results(output_dir):
    #Load the experiment configuration, results, and report from files.  
    # Load configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load results
    results_path = os.path.join(output_dir, "results.csv")
    data = pd.read_csv(results_path)
    
    # Load report if it exists
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
    else:
        report = None
    
    return config, data, report

def llm_classification_experiment(csv_path, client, user_prompt_factory,output_dir, evaluator, true_column="class"):
    results_path = os.path.join(output_dir, "results.csv")
    
    # Check if partial results exist and load them
    if os.path.exists(results_path):
        data = pd.read_csv(results_path)
        start_index = data[data['llm_pred'].isna()].index[0]
        print(start_index)
    else:
        data = pd.read_csv(csv_path, sep=";",encoding='latin1')
        data['llm_pred'] = None
        start_index = 0

    # For each instance in the CSV
    progress_bar = tqdm(data.iterrows(), total=len(data), initial=start_index)
    for index, row in progress_bar:
        if pd.isna(row['llm_pred']):
            user_prompt = user_prompt_factory(row)
            # response would be used for logging, not used right now
            response_text, response = client.generate(user_prompt, progress_bar=progress_bar)
            
            # We add the predicted class to the instance
            data.at[index, 'llm_pred'] = response_text

            # Save the partial results after each API call
            save_experiment_results(output_dir, config, data)

    # Evaluate the predictions
    report = evaluator(data[true_column], data["llm_pred"])
    return data, report

if __name__ == "__main__":
    # Experiment configuration
    config = {
        "csv_path": "../data/dataset_configuration_bug_report_updated.csv",
        "true_column": "Classification",
        "model": "meta-llama-3.1-8b-instruct",
        "system_prompt": """The user will provide information about a bug report. This information includes: Bug-ID, Project, Summary, Description, Link and Enviroment. Classify the bug report into 'Configuration Bug Report' or 'Other', where the first class indicates that the bug report is related or about a configuration bug or issue and the second class indicates that the bug report is NOT RELATED to a configuration bug or issue, and is related to other matters such as database related bugs, request for addition, functional bugs, GUI-related bugs, Network bugs, Performance, security, etc... If the bug report is related to any of those themes, we will classify them as "Other". The first category(Configuration Bug Report) regards bugs concerned with building configuration files. Most of them are related to problems caused by (i) external libraries that should be updated or fixed and (ii) wrong directory or file paths in xml or manifest artifacts. Understand that bug reports can be either 'Configuration Bug Report' or 'Other'. Reply ONLY with one of the two classes, using no more than 3 words: 'Configuration Bug Report' or 'Other'. """,
        "max_tokens": 10,
        "temperature": 0
    }

    config_hash = compute_hash(config)
    output_dir = os.path.join("..", "output", config_hash)
    
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "report.csv")):
        print("Loading existing results...")
        config, data, report = load_experiment_results(output_dir)
    else:
        print("Running new experiment...")
        client = GenerativeModelClient(model=config["model"], system_prompt=config["system_prompt"], max_tokens=config["max_tokens"], temperature=config["temperature"])
        evaluator = edit_distance_evaluator.evaluate
        
        data, report = llm_classification_experiment(config["csv_path"], client, user_prompt_factory.get, output_dir, evaluator, true_column=config["true_column"])
        
        save_experiment_results(config, data, report)
    
    # Results
    print(json.dumps(report, indent=4))
    