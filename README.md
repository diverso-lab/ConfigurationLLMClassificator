# ConfigurationLLMClassificator

This project enables experiments with large language models (LLMs) for classification tasks. It supports processing data using predefined configurations, handling multiple model setups, and generating evaluation reports.


## Features

- **Run Experiments with Configurable Inputs:**  
  Run classification tasks using CSV input files and configuration settings.

- **Support for Investigator and Model Modes:**  
  - Investigator Mode: Execute experiments for a specific investigator using predefined configurations.
  - Models Mode: Execute experiments for multiple models with their respective configurations.

- **Generative Model Integration:**  
  Utilizes LLMs for predictions with user-defined prompts.

- **Partial Result Handling:**  
  Saves intermediate results to prevent data loss during lengthy executions.

- **Evaluation Metrics:**  
  Includes evaluation functionality such as edit distance analysis for classification performance.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/diverso-lab/ConfigurationLLMClassificator
   cd ConfigurationLLMClassificator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Running the Experiment

1. **Investigator Mode:**  
   Execute experiments for a specific investigator using their configuration:
   ```bash
   python main.py --mode i --investigator investigatorName
   ```

2. **Models Mode:**  
   Run experiments for multiple models, **optionally** filtering by specific model names:
   ```bash
   python main.py --mode models --models model1 model2
   ```

## Configuration

### Investigator Configuration
A JSON file (e.g., `configs/investigatorName_config.json`) defines the settings for a single investigator:
```json
{
  "csv_path": "path/to/data.csv",
  "model": "model_name",
  "system_prompt": "Define classification prompt",
  "max_tokens": 256,
  "temperature": 1,
  "true_column": "class"
}
```

### Models Configuration
A JSON file (e.g., `configs/models_config.json`) contains settings for multiple models:
```json
[
  {
    "csv_path": "path/to/data1.csv",
    "model": "model1",
    "system_prompt": "Define prompt",
    "max_tokens": 256,
    "temperature": 1,
    "true_column": "class"
  },
  {
    "csv_path": "path/to/data2.csv",
    "model": "model2",
    "system_prompt": "Define another prompt",
    "max_tokens": 512,
    "temperature": 1,
    "true_column": "label"
  }
]
```

## Output

- **Results Directory:**  
  Results are saved in the `output/` directory with a unique hash based on the configuration.

- **Files:**
  - `config.csv`: Saves the configuration used for this experiment.
  - `results.csv`: Predicted labels for each instance.
  - `report.csv`: Performance metrics and evaluation results.

---