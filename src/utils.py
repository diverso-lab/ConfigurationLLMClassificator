import hashlib
import pandas as pd
import json
import os
from edit_distance_evaluator import edit_distance


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

def clean_predictions(y_true, y_pred):
    unique_values = y_true.unique()
    y_pred = min(unique_values, key=lambda y: edit_distance(y_pred, y))
    return y_pred

# Load a JSON configuration file.
def load_json_config(path):
    with open(path, 'r') as file:
        return json.load(file)
