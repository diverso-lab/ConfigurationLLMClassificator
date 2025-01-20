import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import edit_distance_evaluator
import user_prompt_factory
from GenerativeModelClient import GenerativeModelClient
import json
import utils

def llm_classification_experiment(csv_path, client, user_prompt_factory,output_dir, evaluator,config, true_column="class"):
    results_path = os.path.join(output_dir, "results.csv")
    
    # Check if partial results exist and load them
    if os.path.exists(results_path):
        data = pd.read_csv(results_path)
        if not(data['llm_pred'].isna().any()):
            print("All instances have been classified")
            start_index = data.__len__
        else:
            start_index = data[data['llm_pred'].isna()].index[0]
        print(start_index)
    else:
        data = pd.read_csv(csv_path, sep=";")
        data['llm_pred'] = None
        start_index = 0

    # For each instance in the CSV
    if start_index != data.__len__:
        progress_bar = tqdm(data.iterrows(), total=len(data), initial=start_index)
        for index, row in progress_bar:
            if pd.isna(row['llm_pred']):
                user_prompt = user_prompt_factory(row)
                # response would be used for logging, not used right now
                response_text, response = client.generate(user_prompt, progress_bar=progress_bar)
                
                # We add the predicted class to the instance
                data.at[index, 'llm_pred'] = utils.clean_predictions(data[true_column], response_text)

                # Save the partial results after each API call
                utils.save_experiment_results(output_dir, config, data)

    # Evaluate the predictions
    report = evaluator(data[true_column], data["llm_pred"])
    return data, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM classification experiments.")
    parser.add_argument("--mode", choices=["i", "models"], required=True, help="Execution mode: 'i' for a single investigator configuration, 'models' for running multiple model configurations.")
    parser.add_argument("--investigator", type=str, help="Investigator identifier (e.g., investigator1). Required if mode is 'investigator'.")
    parser.add_argument("--models", nargs="*", help="List of model names to execute. Optional if mode is 'models'.")

    args = parser.parse_args()

    if args.mode == "i":
        if not args.investigator:
            raise ValueError("Investigator identifier is required in 'investigator' mode.")

        investigator_config_path = f"configs/{args.investigator}_config.json"

        if not os.path.exists(investigator_config_path):
            raise FileNotFoundError(f"Configuration file for {args.investigator} not found at {investigator_config_path}")

        investigator_config = utils.load_json_config(investigator_config_path)
        config_hash = utils.compute_hash(investigator_config)
        output_dir = os.path.join("output", config_hash)

        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "report.csv")):
            print("Loading existing results...")
            config, data, report = utils.load_experiment_results(output_dir)

        else:
            print(f"Running experiment for investigator: {args.investigator}")

            client = GenerativeModelClient(model=investigator_config["model"], system_prompt=investigator_config["system_prompt"], max_tokens=investigator_config["max_tokens"], temperature=investigator_config["temperature"])
            evaluator = edit_distance_evaluator.evaluate
            data, report = llm_classification_experiment(investigator_config["csv_path"], client, user_prompt_factory.get, output_dir, evaluator, investigator_config, true_column=investigator_config["true_column"])

            utils.save_experiment_results(output_dir, investigator_config, data, report)

        print("Results:")
        print(json.dumps(report, indent=4))

    elif args.mode == "models":
        models_config_path = "configs/models_config.json"
        if not os.path.exists(models_config_path):
            raise FileNotFoundError(f"Model configurations not found at {models_config_path}")

        models_config = utils.load_json_config(models_config_path)

        # Filter models to execute if specified
        if args.models:
            models_config = [model for model in models_config if model["model"] in args.models]

        for model_config in models_config:
            config_hash = utils.compute_hash(model_config)
            output_dir = os.path.join("output", config_hash)

            if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "report.csv")):
                print("Loading existing results...")
                config, data, report = utils.load_experiment_results(output_dir)

            else:
                print(f"Running experiment for model: {model_config['model']}")

                client = GenerativeModelClient(model=model_config["model"], system_prompt=model_config["system_prompt"], max_tokens=model_config["max_tokens"], temperature=model_config["temperature"])
                evaluator = edit_distance_evaluator.evaluate

                data, report = llm_classification_experiment(model_config["csv_path"], client, user_prompt_factory.get, output_dir, evaluator, model_config, true_column=model_config["true_column"])

                utils.save_experiment_results(output_dir, model_config, data, report)

            print(f"Results for model {model_config['model']}:")
            print(json.dumps(report, indent=4))
    