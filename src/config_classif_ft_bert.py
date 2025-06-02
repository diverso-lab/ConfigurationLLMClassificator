'''
ModernBert, fine-tuning para clasificación binaria
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

def fine_tune_bert(csv_file, title_col="Title", desc_col="Description", class_col="Classification", delim = ",", model_name="answerdotai/ModernBERT-large", batch=8, max_tokens=8192):
    # Load the CSV file
    df = pd.read_csv(csv_file, delimiter=delim)

    #Eliminamos las filas cuya columna class_col contiene una interrogación
    df = df[df[class_col] != '?']

    # Combine title and description into a single input
    df['text'] = "SUMMARY: " + df[title_col] + "\n\n DESCRIPTION:" + df[desc_col]

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[class_col])

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize the data
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=max_tokens)  
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=max_tokens)    

    # Convert class labels to numerical values
    class_labels = {label: idx for idx, label in enumerate(df[class_col].unique())}
    train_labels = torch.tensor(train_df[class_col].map(class_labels).tolist())
    test_labels = torch.tensor(test_df[class_col].map(class_labels).tolist())

    # Mostrar el tamaño del train y el test, y la distribución de clases
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("Class distribution in train set:")
    print(train_df[class_col].value_counts())
    print("Class distribution in test set:")
    print(test_df[class_col].value_counts())
    print("Real proportion of class 1 in test set:", test_df[class_col].value_counts().get('Configuration', 0) / len(test_df))

    # Create a Dataset class
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    # Create train and test datasets
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    # Define training arguments
    train_bsz, val_bsz = batch, batch
    lr = 8e-5
    betas = (0.9, 0.98)
    n_epochs = 10
    eps = 1e-6
    wd = 8e-6
    training_args = TrainingArguments(
        output_dir='./ft_bert_checkpoints',
        learning_rate=lr,
        per_device_train_batch_size=train_bsz,
        per_device_eval_batch_size=val_bsz,
        num_train_epochs=n_epochs,
        lr_scheduler_type="linear",
        weight_decay=wd,
        optim="adamw_torch",
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_epsilon=eps,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
    )

    # Define a compute_metrics function
    eval_results = []

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        macro_f1 = f1_score(p.label_ids, preds, average='macro')
        report = classification_report(p.label_ids, preds, output_dict=True, zero_division=0)
        
        # Calculate the proportion of class 1 in predictions
        pred_class_1_proportion = (preds == 1).sum() / len(preds)
        
        # Calculate the real proportion of class 1 in the test set
        real_class_1_proportion = (p.label_ids == 1).sum() / len(p.label_ids)
        
        # Calculate the absolute error between predicted and real proportions
        abs_error = abs(pred_class_1_proportion - real_class_1_proportion)
        
        # Save the results for later
        eval_results.append({
            'epoch': trainer.state.epoch,
            'f1_macro': macro_f1,
            'f1_class_0': report['0']['f1-score'],
            'f1_class_1': report['1']['f1-score'],
            'pred_class_1_proportion': pred_class_1_proportion,
            'abs_error_class_1_proportion': abs_error
        })
        
        return {
            'macro_f1': macro_f1,
            'f1_class_0': report['0']['f1-score'],
            'f1_class_1': report['1']['f1-score'],
            'pred_class_1_proportion': pred_class_1_proportion,
            'abs_error_class_1_proportion': abs_error
        }

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Convert the evaluation results to a DataFrame and print it
    eval_df = pd.DataFrame(eval_results)
    print(eval_df)


#fine_tune_bert('data/uvl_bug_reports.csv', model_name="answerdotai/ModernBERT-base", batch=2, max_tokens=8192)

#fine_tune_bert('data/dataset_conf_bug_report_v3.csv', title_col="Summary", delim=";", batch=2, max_tokens = 4096, model_name="answerdotai/ModernBERT-base")

fine_tune_bert('data/NABATS_sampled_dataset.csv', title_col="Summary", delim=";", batch=2, max_tokens = 4096, model_name="answerdotai/ModernBERT-base")