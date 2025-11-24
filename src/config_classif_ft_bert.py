'''
ModernBert, fine-tuning para clasificación binaria
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from tqdm.auto import tqdm

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
        bf16=False,
        bf16_full_eval=False,
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

    # Guardar el mejor modelo después del entrenamiento
def predict_with_model(model_dir, csv_file, title_col="Title", desc_col="Description", delim=",",
                           batch=32, max_tokens=8192, label_map=None, output_csv=None, base_model_name="answerdotai/ModernBERT-base"):
        """
        Carga un modelo guardado (model_dir) y hace predicciones sobre csv_file.
        label_map: optional dict mapping label_idx -> label_name (e.g., {0: 'Other', 1: 'Configuration'})
        Devuelve un DataFrame con columnas pred_label, prob_0, prob_1 y pred_label_name (si label_map provisto).
        """
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar datos
        df = pd.read_csv(csv_file, delimiter=delim)
        df[title_col] = df[title_col].fillna("")
        df[desc_col] = df[desc_col].fillna("")
        df['text'] = "SUMMARY: " + df[title_col] + "\n\n DESCRIPTION:" + df[desc_col]
        texts = df['text'].tolist()

        # Cargar tokenizer y modelo guardado
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        all_preds = []
        all_probs = []

        num_texts = len(texts)
        indices = range(0, num_texts, batch)

        for i in tqdm(indices, desc="Generando predicciones", unit="batch"):
            batch_texts = texts[i:i+batch]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_tokens,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc)
                logits = out.logits
                probs = F.softmax(logits, dim=-1).cpu()
                preds = probs.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.numpy().tolist())


        # Añadir resultados al DataFrame
        df['pred_label'] = all_preds
        # Asume problema binario (2 clases); si hay más, ajustar columnas de prob.
        if len(all_probs) > 0 and len(all_probs[0]) >= 1:
            for idx in range(len(all_probs[0])):
                df[f'prob_{idx}'] = [p[idx] for p in all_probs]

        if label_map is not None:
            inv_map = {int(k): v for k, v in label_map.items()}
            df['pred_label_name'] = df['pred_label'].map(inv_map)

        if output_csv:
            df_out = df.drop(columns=['text'])
            df.to_csv(output_csv, index=False)

        print("Predicciones completadas. Proporción de clase 1 predicha:",
            (df['pred_label'] == 1).mean() if 'pred_label' in df else None)

        return df


df_preds = predict_with_model("ft_bert_checkpoints/checkpoint-6810", "data/TID.csv",
                                 title_col="Summary", desc_col="Description", delim=";",
                                 batch=2, max_tokens=8192,
                                 label_map={0: "Other", 1: "Configuration"},
                                 output_csv="output/predsTID.csv")


#fine_tune_bert('data/uvl_bug_reports.csv', model_name="answerdotai/ModernBERT-base", batch=2, max_tokens=8192)

#fine_tune_bert('data/dataset_conf_bug_report_v3.csv', title_col="Summary", delim=";", batch=2, max_tokens = 4096, model_name="answerdotai/ModernBERT-base")

#fine_tune_bert('data/misconfiguration_datasets.csv', title_col="Summary", delim=";", batch=2, max_tokens = 4096, model_name="answerdotai/ModernBERT-base")

#fine_tune_bert('data/grouped_dataset.csv', title_col="Summary", delim=";", batch=2, max_tokens = 4096, model_name="answerdotai/ModernBERT-base")